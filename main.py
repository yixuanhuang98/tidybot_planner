# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
from policies import TeleopPolicy, RemotePolicy, MotionPlannerPolicy
from policies import MotionPlannerPolicyStackWrapper
from policies import MotionPlannerPolicyStackThreeWrapper

def should_save_episode(writer):
    if len(writer) == 0:
        print('Discarding empty episode')
        return False

    # Prompt user whether to save episode
    while True:
        user_input = input('Save episode (y/n)? ').strip().lower()
        if user_input == 'y':
            return True
        if user_input == 'n':
            print('Discarding episode')
            return False
        print('Invalid response')

def run_episode(env, policy, writer=None):
    # Reset the env
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    # Wait for user to press "Start episode"
    print('Press "Start episode" in the web app when ready to start new episode')
    policy.reset()
    print('Starting new episode')

    episode_ended = False
    start_time = time.time()
    for step_idx in count():
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # Get latest observation
        obs = env.get_obs()

        # Get action
        action = policy.step(obs)
        # print('action', action)

        # No action if teleop not enabled
        if action is None:
            continue

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

            if writer is not None and should_save_episode(writer):
                # Save to disk in background thread
                writer.flush_async()

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
        from mujoco_env import MujocoEnv
        if args.drawer_scene:
            env = MujocoEnv(drawer_scene=True)
        elif args.table_scene:
            env = MujocoEnv(table_scene=True)
        elif args.cupboard_scene:
            env = MujocoEnv(cupboard_scene=True)
        elif args.teleop:
            env = MujocoEnv(show_images=True)
        else:
            env = MujocoEnv()
    else:
        from real_env import RealEnv
        env = RealEnv()

    # Create policy
    if args.drawer_scene and args.stack_policy_three:
        from policies import MotionPlannerPolicyStackDrawerThreeWrapper
        policy = MotionPlannerPolicyStackDrawerThreeWrapper()
    elif args.drawer_scene and args.stack_policy:
        from policies import MotionPlannerPolicyStackDrawerWrapper
        policy = MotionPlannerPolicyStackDrawerWrapper()
    elif args.table_scene and args.stack_policy_three:
        from policies import MotionPlannerPolicyStackTableThreeWrapper
        policy = MotionPlannerPolicyStackTableThreeWrapper()
    elif args.table_scene and args.stack_policy:
        from policies import MotionPlannerPolicyStackTableWrapper
        policy = MotionPlannerPolicyStackTableWrapper()
    elif args.stack_policy_three:
        from policies import MotionPlannerPolicyStackThreeWrapper
        policy = MotionPlannerPolicyStackThreeWrapper()
    elif args.stack_policy:
        from policies import MotionPlannerPolicyStackWrapper
        policy = MotionPlannerPolicyStackWrapper()
    elif args.motion_planner:
        from policies import MotionPlannerPolicy
        policy = MotionPlannerPolicy()
    elif args.mp_policy:
        from policies import MotionPlannerPolicyMPWrapper
        policy = MotionPlannerPolicyMPWrapper()
    elif args.teleop:
        from policies import TeleopPolicy
        policy = TeleopPolicy()
    else:
        from policies import RemotePolicy
        policy = RemotePolicy()

    try:
        while True:
            writer = EpisodeWriter(args.output_dir) if args.save else None
            run_episode(env, policy, writer)
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--teleop', action='store_true')
    parser.add_argument('--motion_planner', action='store_true')
    parser.add_argument('--mp_policy', action='store_true', help='Enable new motion planner policy from agent/mp_policy.py')
    parser.add_argument('--stack_policy', action='store_true', help='Enable stacking policy (stack cubes)')
    parser.add_argument('--stack_policy_three', action='store_true', help='Enable stacking policy (stack three cubes) - works with both ground and table scenes')
    parser.add_argument('--table_scene', action='store_true', help='Use table scene with three cubes')
    parser.add_argument('--drawer_scene', action='store_true', help='Use drawer scene with three cubes')
    parser.add_argument('--cupboard_scene', action='store_true', help='Use cupboard scene with three cubes')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output-dir', default='data/demos')
    main(parser.parse_args())
