# Author: Jimmy Wu
# Date: October 2024
# for replayrobomimic hdf5 files

import argparse
import time
from itertools import count
from pathlib import Path
import cv2 as cv
import h5py
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeReader
from mujoco_env import MujocoEnv
from scipy.spatial.transform import Rotation

def replay_episode(env, input_file, show_images=False, execute_obs=False):
    # Reset env
    

    # Load episode data
    input_file = h5py.File(input_file, 'r')
    print(f'Loaded episodes from {input_file}')
    input_file_group = input_file['data']
    for episode_key in input_file_group.keys():
        env.reset()
        episode_group = input_file_group[episode_key]
        observations = episode_group['obs']
        actions = episode_group['actions']
        language = episode_group['language']
        print(f'Loaded episode {episode_key} from {input_file}')

        start_time = time.time()
        import pdb; pdb.set_trace()
        for step_idx in range(len(observations['base_pose'])):
            obs = {
                'base_pose': observations['base_pose'][step_idx],
                'arm_pos': observations['arm_pos'][step_idx],
                'arm_quat': observations['arm_quat'][step_idx],
                'gripper_pos': observations['gripper_pos'][step_idx],
            }
            if len(actions[step_idx]) == 11:
                action = {
                    'base_pose': actions[step_idx][:3],
                    'arm_pos': actions[step_idx][3:6],
                    'arm_quat': actions[step_idx][6:10],
                    'gripper_pos': actions[step_idx][10:11],
                }
            else:
                action = {
                    'base_pose': actions[step_idx][:3],
                    'arm_pos': actions[step_idx][3:6],
                    'arm_quat': Rotation.from_rotvec(actions[step_idx][6:9]).as_quat(),
                    'gripper_pos': actions[step_idx][9:10],
                }
            # Enforce desired control freq
            step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
            while time.time() < step_end_time:
                time.sleep(0.0001)

            # Show image observations
            if show_images:
                window_idx = 0
                for k, v in obs.items():
                    if v.ndim == 3:
                        cv.imshow(k, cv.cvtColor(v, cv.COLOR_RGB2BGR))
                        cv.moveWindow(k, 640 * window_idx, 0)
                        window_idx += 1
                cv.waitKey(1)

            # Execute in action in env
            if execute_obs:
                env.step(obs)
            else:
                env.step(action)

def main(args):
    # Create env
    if args.sim:
        env = MujocoEnv(render_images=False)
    else:
        from real_env import RealEnv
        env = RealEnv()

    try:
        replay_episode(env, args.input_file, show_images=args.show_images, execute_obs=args.execute_obs)
            # input('Press <Enter> to continue...')
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', default='data/demos.hdf5')
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--show-images', action='store_true')
    parser.add_argument('--execute-obs', action='store_true')
    main(parser.parse_args())
