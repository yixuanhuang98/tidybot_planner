# Author: Jimmy Wu (original), Modified for residual learning
# Date: October 2024
#
# Converts episode data to robomimic HDF5 format with residual learning support.
# - Adds planning_action to observations (13D with rotation_6d, matches model output)
# - Computes delta_action using proper rotation difference (10D with axis_angle)
# - For rotation: R_delta = R_gt * R_plan.inv() (mathematically correct)
# - For non-rotation parts: direct subtraction
# - At inference: final_action = planning_action (13D) + predicted_delta (13D)

import argparse
from pathlib import Path
import cv2 as cv
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from constants import POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT
from episode_storage import EpisodeReader


def quat_to_rotation_6d(quat):
    """Convert quaternion to 6D rotation representation.
    
    Uses pytorch3d convention: first two ROWS of rotation matrix, flattened.
    """
    rot_matrix = Rotation.from_quat(quat).as_matrix()
    rot_6d = rot_matrix[:2, :].flatten()  # [r00,r01,r02,r10,r11,r12]
    return rot_6d


def action_dict_to_array_10d(action):
    """Convert action dict to numpy array [10] with axis_angle (rotvec)"""
    return np.concatenate((
        action['base_pose'],                                    # 3
        action['arm_pos'],                                      # 3
        Rotation.from_quat(action['arm_quat']).as_rotvec(),    # 3 (axis_angle)
        action['gripper_pos'],                                  # 1
    ))                                                         # = 10


def action_dict_to_array_13d(action):
    """Convert action dict to numpy array [13] with rotation_6d"""
    return np.concatenate((
        action['base_pose'],                        # 3
        action['arm_pos'],                          # 3
        quat_to_rotation_6d(action['arm_quat']),   # 6 (rotation_6d)
        action['gripper_pos'],                      # 1
    ))                                             # = 13


def compute_delta_action(gt_action, planning_action):
    """Compute delta action with proper rotation difference.
    
    For non-rotation parts: direct subtraction
    For rotation part: R_delta = R_gt * R_plan.inv() (mathematically correct)
    
    Args:
        gt_action: dict with base_pose, arm_pos, arm_quat, gripper_pos
        planning_action: dict with same keys
    
    Returns:
        numpy array [10] - delta action with axis_angle rotation
    """
    # Non-rotation parts: direct subtraction
    delta_base_pose = gt_action['base_pose'] - planning_action['base_pose']  # 3D
    delta_arm_pos = gt_action['arm_pos'] - planning_action['arm_pos']        # 3D
    delta_gripper = gt_action['gripper_pos'] - planning_action['gripper_pos']  # 1D
    
    # Rotation part: compute proper rotation difference
    # R_delta = R_gt * R_plan.inv()
    # This gives the rotation that, when composed with R_plan, gives R_gt
    R_gt = Rotation.from_quat(gt_action['arm_quat'])
    R_plan = Rotation.from_quat(planning_action['arm_quat'])
    R_delta = R_gt * R_plan.inv()
    delta_rotvec = R_delta.as_rotvec()  # 3D
    
    return np.concatenate([
        delta_base_pose,   # 3
        delta_arm_pos,     # 3
        delta_rotvec,      # 3
        delta_gripper,     # 1
    ])                    # = 10


def _get_cube_keys(observation):
    return sorted([k for k in observation.keys() if k.startswith('cube') and k.endswith('_pos')])


def compute_residual_mask(
    observations,
    ee_cube_dist_thresh=0.11,
    cube_motion_thresh=0.003,
    ee_cube_top_z_thresh=0.0,
):
    """Mask residual learning to contact/push-relevant windows.

    A step is active if either:
    - end-effector is near any cube in XY plane, or
    - any cube moves noticeably between current and next observation.
    """
    num_steps = len(observations)
    if num_steps == 0:
        return np.zeros((0,), dtype=np.float32)

    cube_keys = _get_cube_keys(observations[0])
    mask = np.zeros((num_steps,), dtype=np.float32)
    for t in range(num_steps):
        obs_t = observations[t]
        arm_pos = obs_t.get('arm_pos', None)

        near_cube = False
        if arm_pos is not None and cube_keys:
            arm_xy = np.asarray(arm_pos[:2], dtype=np.float64)
            ee_z = float(np.asarray(arm_pos, dtype=np.float64)[2])

            cube_dists = []
            z_diffs = []
            for cube_key in cube_keys:
                if cube_key in obs_t:
                    cube_pos = np.asarray(obs_t[cube_key], dtype=np.float64)
                    cube_xy = cube_pos[:2]
                    cube_dists.append(np.linalg.norm(arm_xy - cube_xy))
                    # Approximate cube top from center z. In this env cube center z ~= cube half-size.
                    cube_top = 2.0 * float(cube_pos[2])
                    z_diffs.append(ee_z - cube_top)

            # Only consider XY-nearest cube when EE is below all cube tops.
            if cube_dists and z_diffs and max(z_diffs) < ee_cube_top_z_thresh:
                near_cube = min(cube_dists) <= ee_cube_dist_thresh

        cube_moving = False
        if t < num_steps - 1 and cube_keys:
            obs_tp1 = observations[t + 1]
            cube_motion = []
            for cube_key in cube_keys:
                if cube_key in obs_t and cube_key in obs_tp1:
                    dxy = np.asarray(obs_tp1[cube_key][:2], dtype=np.float64) - np.asarray(obs_t[cube_key][:2], dtype=np.float64)
                    cube_motion.append(np.linalg.norm(dxy))
            if cube_motion:
                cube_moving = max(cube_motion) >= cube_motion_thresh

        if near_cube or cube_moving:
            mask[t] = 1.0
    return mask


def main(
    input_dir,
    output_path,
    ee_cube_dist_thresh,
    cube_motion_thresh,
    ee_cube_top_z_thresh,
    zero_delta_outside_mask,
):
    # Get list of episode dirs
    episode_dirs = sorted([child for child in Path(input_dir).iterdir() if child.is_dir()])

    if len(episode_dirs) == 0:
        print(f"No episode directories found in {input_dir}")
        return

    # Convert to robomimic HDF5 format
    with h5py.File(output_path, 'w') as f:
        data_group = f.create_group('data')
        valid_episode_count = 0

        # Iterate through episodes
        for episode_idx, episode_dir in enumerate(tqdm(episode_dirs)):
            # Load episode data
            try:
                reader = EpisodeReader(episode_dir)
            except AssertionError as exc:
                print(f"Warning: malformed episode {episode_dir}, skipping... ({exc})")
                continue

            # Check if planning_actions exist
            has_planning = reader.planning_actions is not None
            if not has_planning:
                print(f"Warning: {episode_dir} has no planning_actions, skipping...")
                continue

            num_steps = min(len(reader.observations), len(reader.actions), len(reader.planning_actions))
            if num_steps == 0:
                print(f"Warning: {episode_dir} has empty data, skipping...")
                continue

            # Extract observations
            observations = {}
            for obs in reader.observations[:num_steps]:
                for k, v in obs.items():
                    if v.ndim == 3:
                        # Resize image
                        v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))

                    # Append extracted observation
                    if k not in observations:
                        observations[k] = []
                    observations[k].append(v)

            # Extract planning_actions (13D with rotation_6d) and add to observations
            # Using 13D so it matches the model output dimension at inference time
            planning_actions = [action_dict_to_array_13d(pa) for pa in reader.planning_actions[:num_steps]]
            observations['planning_action'] = planning_actions
            residual_mask = compute_residual_mask(
                reader.observations[:num_steps],
                ee_cube_dist_thresh=ee_cube_dist_thresh,
                cube_motion_thresh=cube_motion_thresh,
                ee_cube_top_z_thresh=ee_cube_top_z_thresh,
            )
            observations['residual_mask'] = residual_mask

            # Compute delta_action with proper rotation difference
            # Non-rotation parts: direct subtraction
            # Rotation part: R_delta = R_gt * R_plan.inv()
            delta_actions = [
                compute_delta_action(gt, pa) 
                for gt, pa in zip(reader.actions[:num_steps], reader.planning_actions[:num_steps])
            ]
            if zero_delta_outside_mask:
                delta_actions = [
                    delta if residual_mask[i] > 0.5 else np.zeros_like(delta)
                    for i, delta in enumerate(delta_actions)
                ]

            # Write to HDF5
            episode_key = f'demo_{valid_episode_count}'
            episode_group = data_group.create_group(episode_key)
            for k, v in observations.items():
                episode_group.create_dataset(f'obs/{k}', data=np.array(v))
            episode_group.create_dataset('actions', data=np.array(delta_actions))
            episode_group.attrs['residual_active_ratio'] = float(np.mean(residual_mask))
            valid_episode_count += 1

        print(f"\nSaved to {output_path}")
        print(f"Total episodes: {valid_episode_count}")
        print(f"Action (delta) dimension: 10 (axis_angle, auto-converted to 13 during training)")
        print(f"Planning_action dimension: 13 (rotation_6d, same as model output)")
        print("Added obs/residual_mask: 1=contact/push window, 0=outside window")
        print(f"Rotation delta: R_delta = R_gt * R_plan.inv() (mathematically correct)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/residual_demos')
    parser.add_argument('--output-path', default='data/residual_demos.hdf5')
    parser.add_argument('--ee-cube-dist-thresh', type=float, default=0.11)
    parser.add_argument('--cube-motion-thresh', type=float, default=0.003)
    parser.add_argument('--ee-cube-top-z-thresh', type=float, default=0.0)
    parser.add_argument('--zero-delta-outside-mask', action='store_true')
    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_path,
        args.ee_cube_dist_thresh,
        args.cube_motion_thresh,
        args.ee_cube_top_z_thresh,
        args.zero_delta_outside_mask,
    )
