# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
from itertools import count
from pathlib import Path
import cv2 as cv
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeReader
from mujoco_env import MujocoEnv

def replay_episode(env, episode_dir, show_images=False, execute_obs=False):
    # Reset env
    env.reset()

    # Load episode data
    reader = EpisodeReader(episode_dir)
    print(f'Loaded episode from {episode_dir}')

    if hasattr(reader, 'initial_sim_state') and reader.initial_sim_state is not None and hasattr(env, 'set_sim_state'):
        env.set_sim_state(reader.initial_sim_state)

    start_time = time.time()
    for step_idx, (obs, action) in enumerate(zip(reader.observations, reader.actions)):
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
        episode_dirs = sorted([child for child in Path(args.input_dir).iterdir() if child.is_dir()])
        for episode_dir in episode_dirs:
            replay_episode(env, episode_dir, show_images=args.show_images, execute_obs=args.execute_obs)
            # input('Press <Enter> to continue...')
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/demos')
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--show-images', action='store_true')
    parser.add_argument('--execute-obs', action='store_true')
    main(parser.parse_args())
