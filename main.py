# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
# MotionPlannerPolicy: task uses MOTION_PLANNER_NUM_TASK_CUBES in policies.py (2 vs 3), or
# MotionPlannerPolicy(num_task_cubes=3) here for a one-off override.
from policies import TeleopPolicy, RemotePolicy, MotionPlannerPolicy, ResidualPolicy

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

def run_episode(env, policy, writer=None, args=None, planner=None):
    # Reset the env
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    if writer is not None and args is not None and args.sim and hasattr(env, 'get_sim_state'):
        writer.set_initial_sim_state(env.get_sim_state())

    # Wait for user to press "Start episode"
    print('Press "Start episode" in the web app when ready to start new episode')
    policy.reset()
    if planner is not None:
        planner.reset()
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

        # Get action from main policy (teleop or other)
        action = policy.step(obs)
        # print('action', action)
        # Get planning action if in residual mode
        planning_action = None
        if planner is not None and isinstance(action, dict):
            planning_action = planner._step(obs)

        # No action if teleop not enabled
        if action is None:
            continue

        if step_idx > 3000:
            break # avoid infinite loop
        
        # Execute valid action on robot
        if isinstance(action, dict):
            env.step(action)

            if writer is not None and not episode_ended:
                # Record executed action
                writer.step(obs, action, planning_action)

        # Episode ended
        elif not episode_ended and action == 'end_episode':
            episode_ended = True
            print('Episode ended')
            print('step_idx', step_idx)

            if writer is not None and should_save_episode(writer, args):
                # Save to disk in background thread
                writer.flush_async()

            if args.sim and (args.motion_planner or args.residual_infer):
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
        from mujoco_env import MujocoEnv
        if args.teleop or args.residual:
            env = MujocoEnv(show_images=True)
        else:
            env = MujocoEnv()
    else:
        from real_env import RealEnv
        env = RealEnv()

    # Create policy
    if args.motion_planner:
        policy = MotionPlannerPolicy()
    elif args.residual_infer:
        policy = ResidualPolicy(enable_web_server=not args.sim)
    elif args.teleop or args.residual:
        policy = TeleopPolicy()
    else:
        policy = RemotePolicy(enable_web_server=not args.sim)

    if args.residual:
        print(f"\n{'='*50}"+'\nRunning residual mode\n'+f"{'='*50}\n")
        planner = MotionPlannerPolicy()
    else:
        planner = None
    NUM_EPISODES = 1 # Change this to run more/fewer episodes
    try:
        for episode in range(NUM_EPISODES):
            print(f"\n{'='*50}\nEPISODE {episode + 1}/{NUM_EPISODES}\n{'='*50}")
            writer = EpisodeWriter(args.output_dir, save_planning_action=args.residual) if args.save else None
            run_episode(env, policy, writer, args, planner=planner)
    finally:
        if args.motion_planner:
            policy.print_final_stats()
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--teleop', action='store_true')
    parser.add_argument('--motion_planner', action='store_true')
    parser.add_argument('--residual', action='store_true', help='Residual data collection mode: teleop + record planner output')
    parser.add_argument('--residual_infer', action='store_true', help='Residual inference mode: planner + diffusion delta')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output-dir', default='data/demos')
    main(parser.parse_args())
