# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
from itertools import count
from pathlib import Path
from datetime import datetime
import cv2 as cv
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeReader
from mujoco_env import MujocoEnv

def replay_episode(env, episode_dir, show_images=False, execute_obs=False, save_images=False, output_dir=None):
    # Reset env
    env.reset()

    # Load episode data
    reader = EpisodeReader(episode_dir)
    print(f'Loaded episode from {episode_dir}')

    # Setup for image saving
    if save_images and output_dir:
        episode_name = episode_dir.name
        episode_output_dir = output_dir / f'replay_{episode_name}'
        episode_output_dir.mkdir(parents=True, exist_ok=True)
        print(f'Saving episode images to: {episode_output_dir}')

    start_time = time.time()
    for step_idx, (obs, action) in enumerate(zip(reader.observations, reader.actions)):
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # Show or save image observations
        if show_images or save_images:
            window_idx = 0
            for k, v in obs.items():
                if v.ndim == 3:
                    if show_images:
                        cv.imshow(k, cv.cvtColor(v, cv.COLOR_RGB2BGR))
                        cv.moveWindow(k, 640 * window_idx, 0)
                        window_idx += 1
                    
                    if save_images and output_dir:
                        timestamp = f'step_{step_idx:06d}'
                        image_path = episode_output_dir / f'{k}_{timestamp}.jpg'
                        cv.imwrite(str(image_path), cv.cvtColor(v, cv.COLOR_RGB2BGR))
            
            if show_images:
                cv.waitKey(1)

        # Execute in action in env
        if execute_obs:
            env.step(obs)
        else:
            env.step(action)

    if save_images and output_dir:
        print(f'Completed saving images for episode {episode_dir.name}')

def main(args):
    # Create output directory for saved images if needed
    output_dir = None
    if args.save_images:
        output_dir = Path('replay_images') / datetime.now().strftime('%Y%m%dT%H%M%S')
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Episode replay images will be saved to: {output_dir}")

    # Create env
    if args.sim:
        # Enable rendering only if we need to show or save images
        render_images = args.show_images or args.save_images
        env = MujocoEnv(render_images=render_images, show_viewer=False, show_images=False, save_images=False)
    else:
        from real_env import RealEnv
        env = RealEnv()

    try:
        episode_dirs = sorted([child for child in Path(args.input_dir).iterdir() if child.is_dir()])
        for episode_dir in episode_dirs:
            replay_episode(env, episode_dir, show_images=args.show_images, execute_obs=args.execute_obs, 
                          save_images=args.save_images, output_dir=output_dir)
            # input('Press <Enter> to continue...')
        
        if args.save_images:
            print(f"\nAll episode images saved to: {output_dir}")
            
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/demos')
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--show-images', action='store_true')
    parser.add_argument('--save-images', action='store_true', help='Save episode images to disk instead of displaying')
    parser.add_argument('--execute-obs', action='store_true')
    main(parser.parse_args())
