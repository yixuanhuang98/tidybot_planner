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


def main(input_dir, output_path):
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

            # Extract observations
            observations = {}
            for obs in reader.observations:
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
            planning_actions = [action_dict_to_array_13d(pa) for pa in reader.planning_actions]
            observations['planning_action'] = planning_actions

            # Compute delta_action with proper rotation difference
            # Non-rotation parts: direct subtraction
            # Rotation part: R_delta = R_gt * R_plan.inv()
            delta_actions = [
                compute_delta_action(gt, pa) 
                for gt, pa in zip(reader.actions, reader.planning_actions)
            ]

            # Write to HDF5
            episode_key = f'demo_{valid_episode_count}'
            episode_group = data_group.create_group(episode_key)
            for k, v in observations.items():
                episode_group.create_dataset(f'obs/{k}', data=np.array(v))
            episode_group.create_dataset('actions', data=np.array(delta_actions))
            valid_episode_count += 1

        print(f"\nSaved to {output_path}")
        print(f"Total episodes: {valid_episode_count}")
        print(f"Action (delta) dimension: 10 (axis_angle, auto-converted to 13 during training)")
        print(f"Planning_action dimension: 13 (rotation_6d, same as model output)")
        print(f"Rotation delta: R_delta = R_gt * R_plan.inv() (mathematically correct)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/residual_demos')
    parser.add_argument('--output-path', default='data/residual_demos.hdf5')
    args = parser.parse_args()
    main(args.input_dir, args.output_path)
