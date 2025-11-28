# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
import random
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
from policies import TeleopPolicy, RemotePolicy, MotionPlannerPolicy

def should_save_episode(writer, args):
    if len(writer) == 0:
        print('Discarding empty episode')
        return False

    # Prompt user whether to save episode
    if args.sim and args.motion_planner:
        return True
    else:
        while True:
            user_input = input('Save episode (y/n)? ').strip().lower()
            if user_input == 'y':
                return True
            if user_input == 'n':
                print('Discarding episode')
                return False
            print('Invalid response')

def run_episode(env, policy, writer=None, args=None):
    # Reset the env
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    # Wait for user to press "Start episode"
    print('Press "Start episode" in the web app when ready to start new episode')
    target_object_key = random.choice(['cake', 'apple', 'bar'])
    policy.reset(target_object_key)
    print('Starting new episode')
    print('Target object key: ', target_object_key)

    episode_ended = False
    start_time = time.time()
    success = False
    for step_idx in count():
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        
        # Get latest observation
        obs = env.get_obs()
        
        
        for key in obs:
            if target_object_key in key and 'pos' in key:
                pos = obs[key]
                if pos[2] > 0.1:
                    success = True
                    break
        if success:
            break
                    

        # Get action
        action = policy.step(obs)
        # print('action', action)

        # No action if teleop not enabled
        if action is None:
            continue

        if step_idx > 50:
            break # avoid infinite loop
        
        # Execute valid action on robot
        if isinstance(action, dict):
            env.step(action)

            if writer is not None and not episode_ended:
                # Record executed action
                writer.step(obs, action)

        # Episode ended
        elif not episode_ended and action == 'end_episode':
            episode_ended = True
            print('Episode ended')
            print('step_idx', step_idx)

            if writer is not None and should_save_episode(writer, args):
                # Save to disk in background thread
                writer.flush_async()

            if args.sim and args.motion_planner:
                print('Episode ended')
                break
            else:
                print('Teleop is now active. Press "Reset env" in the web app when ready to proceed.')

        
        # Ready for env reset
        elif action == 'reset_env':
            break

    if writer is not None:
        # Wait for writer to finish saving to disk
        writer.wait_for_flush()

def main(args):
    # Create env
    if args.sim:
        # Use customizable environment if specified
        if args.custom_env:
            from customizable_env import CustomizableMujocoEnv
            from env_configs import get_config, create_random_config

            # Load configuration
            if args.env_config:
                # Use predefined config
                env_config = get_config(args.env_config)
            elif args.random_env:
                # Create random config
                env_config = create_random_config(
                    floor=args.floor_texture,
                    object_category=args.object_category,
                    num_objects=args.num_objects,
                    headless=(not args.teleop)
                )
            else:
                # Custom config from command line
                env_config = {
                    'floor_texture': args.floor_texture,
                    'objects': args.objects.split(',') if args.objects else ['apple.glb', 'banana.glb', 'tomato.glb'],
                    'render_images': True,
                    'show_viewer': not args.render_only,  # Hide viewer in render-only mode
                    'show_images': args.teleop
                }

            env = CustomizableMujocoEnv(**env_config)

            # Offline rendering mode
            if args.render_only:
                print("\n=== Offline Rendering Mode ===")
                try:
                    env.render_first_frame(
                        output_dir=args.render_output_dir,
                        prefix=args.render_prefix,
                        save_state=True
                    )
                    print("\nRendering complete! Exiting...")
                finally:
                    env.close()
                return  # Exit after rendering
        else:
            # Use default MujocoEnv
            from mujoco_env import MujocoEnv
            if args.teleop:
                env = MujocoEnv(show_images=True)
            else:
                env = MujocoEnv()
    else:
        from real_env import RealEnv
        env = RealEnv()

    # Create policy
    if args.motion_planner:
        policy = MotionPlannerPolicy()
    elif args.teleop:
        policy = TeleopPolicy()
    else:
        policy = RemotePolicy(enable_web_server=not args.sim)

    try:
        while True:
            writer = EpisodeWriter(args.output_dir) if args.save else None
            run_episode(env, policy, writer, args)
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true', help='Use simulation environment')
    parser.add_argument('--teleop', action='store_true', help='Enable teleoperation')
    parser.add_argument('--motion_planner', action='store_true', help='Use motion planner policy')
    parser.add_argument('--save', action='store_true', help='Save episodes to disk')
    parser.add_argument('--output-dir', default='data/demos', help='Directory to save episodes')

    # Customizable environment options
    parser.add_argument('--custom-env', action='store_true',
                       help='Use customizable environment with custom objects and textures')
    parser.add_argument('--env-config', type=str, default=None,
                       help='Predefined environment config name (e.g., pick_place_fruits, sorting_mixed)')
    parser.add_argument('--random-env', action='store_true',
                       help='Use randomly generated environment configuration')
    parser.add_argument('--floor-texture', type=str, default=None,
                       help='Floor texture name or path (e.g., light_wood_v3.png)')
    parser.add_argument('--objects', type=str, default=None,
                       help='Comma-separated list of objects (e.g., apple.glb,banana.glb,tomato.glb)')
    parser.add_argument('--object-category', type=str, default='fruits',
                       choices=['fruits', 'vegetables', 'containers', 'tools',
                               'robocasa_fruits', 'robocasa_vegetables', 'robocasa_containers', "easygrasp_objects"],
                       help='Object category for random environment generation')
    parser.add_argument('--num-objects', type=int, default=3,
                       help='Number of objects for random environment generation')

    # Offline rendering options
    parser.add_argument('--render-only', action='store_true',
                       help='Offline rendering mode: render first frame and exit (requires --custom-env)')
    parser.add_argument('--render-output-dir', type=str, default='env_renders',
                       help='Directory to save rendered images')
    parser.add_argument('--render-prefix', type=str, default='env',
                       help='Prefix for rendered image filenames')

    main(parser.parse_args())
