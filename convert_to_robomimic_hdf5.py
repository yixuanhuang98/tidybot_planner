# Author: Jimmy Wu
# Date: October 2024
#
# References:
# - https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/dataset_states_to_obs.py

import argparse
from pathlib import Path
import cv2 as cv
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from constants import POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT
from episode_storage import EpisodeReader

def main(input_dir, output_path, args):
    # Get list of episode dirs
    episode_dirs = sorted([child for child in Path(input_dir).iterdir() if child.is_dir()])

    # Convert to robomimic HDF5 format
    with h5py.File(output_path, 'w') as f:
        data_group = f.create_group('data')

        # Iterate through episodes
        for episode_idx, episode_dir in enumerate(tqdm(episode_dirs)):
            # Load episode data
            if episode_idx >= args.max_episodes:
                break
            reader = EpisodeReader(episode_dir)

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

            # Extract actions
            if args.follow_obs:
                actions = []
                gripper_pos = 0.0
                for i in range(len(reader.actions)):
                    if i == len(reader.actions) - 1:
                        actions.append(np.concatenate((
                            reader.actions[i]['base_pose'],
                            reader.actions[i]['arm_pos'],
                            reader.actions[i]['arm_quat'],
                            reader.actions[i]['gripper_pos'],
                        )))
                    else:
                        # import pdb; pdb.set_trace()
                        if reader.actions[i]['gripper_pos'] == 1.0:
                            gripper_pos = 1.0
                        actions.append(np.concatenate((
                            observations['base_pose'][i+1],
                            observations['arm_pos'][i+1],
                            observations['arm_quat'][i+1],
                            np.array([gripper_pos]),
                        )))
            elif args.quaternion:
                actions = [
                    np.concatenate((
                        action['base_pose'],
                        action['arm_pos'],
                        action['arm_quat'],  # Convert quat to axis-angle
                        action['gripper_pos'],
                    )) for action in reader.actions
                ]
            else:  # Convert quat to axis-angle
                actions = [
                    np.concatenate((
                        action['base_pose'],
                        action['arm_pos'],
                        Rotation.from_quat(action['arm_quat']).as_rotvec(),  # Convert quat to axis-angle
                        action['gripper_pos'],
                    )) for action in reader.actions
                ]

            
            
            if args.predicate:
                predicates = []
                state = 'moving'
                for t in range(len(reader.observations)):
                    # import pdb; pdb.set_trace()
                    if state == 'moving':
                        if np.linalg.norm(reader.observations[t+1]['base_pose'] - reader.observations[t]['base_pose']) > 0.001:
                            predicates.append("The target object in front of the base")
                        else:
                            predicates.append("The target object is in the gripper")
                            state = 'reach_moving_target'
                    elif state == 'reach_moving_target':
                        if reader.observations[t]['gripper_pos'] > 0.3 and (reader.observations[t+1]['gripper_pos'] - reader.observations[t]['gripper_pos'] < 0.01):
                            predicates.append("The target object is on the ground")
                            state = 'moving_object'
                        else:
                            predicates.append("The target object is in the gripper")
                    elif state == 'moving_object':
                        predicates.append("The target object is on the ground")
            
            # Write to HDF5
            episode_key = f'demo_{episode_idx}'
            episode_group = data_group.create_group(episode_key)
            for k, v in observations.items():
                episode_group.create_dataset(f'obs/{k}', data=np.array(v))
            print('actions', actions)
            episode_group.create_dataset('actions', data=np.array(actions))
            if args.language:
                if len(reader.target_object_key) > 0:
                    target_object_key = reader.target_object_key[0]
                    target_object_key = target_object_key.split('_')[0]
                    episode_group.create_dataset('language', data=f"Pick the {target_object_key} and place it in the +x direction by 0.5m.")
                else:
                    episode_group.create_dataset('language', data="Pick the target object and place it in the +x direction by 0.5m.")
            if args.predicate:
                assert len(predicates) == len(reader.actions)
                assert len(predicates) == len(reader.observations)
                # print('actions', reader.actions)
                episode_group.create_dataset('predicates', data=predicates)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/demos')
    parser.add_argument('--output-path', default='data/demos.hdf5')
    parser.add_argument('--language', type=bool, default=False)
    parser.add_argument('--predicate', type=bool, default=False)
    parser.add_argument('--quaternion', type=bool, default=False)
    parser.add_argument('--follow_obs', type=bool, default=False)
    parser.add_argument('--max_episodes', type=int, default=1000000)
    args = parser.parse_args()
    main(args.input_dir, args.output_path, args = args)