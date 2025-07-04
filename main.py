# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from agent.teleop_policy import TeleopPolicy
from agent.remote_policy import RemotePolicy
from agent.motion_planner_policy import MotionPlannerPolicy
from agent.go_to_cabinet_handle_policy import GoToCabinetHandlePolicy
from agent.go_to_cabinet_handle_policy_right import GoToCabinetHandlePolicyRight
# from agent.motion_planner_policy_stack import MotionPlannerPolicyStack

def run_episode(env, policy):
    # Reset the env
    print('Resetting env...')
    obs, info = env.reset()
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
        # while time.time() < step_end_time:
        #     time.sleep(0.0000001)

        # Get action
        action = policy.step(obs)
        # print('action', action)

        # No action: either teleop disabled or policy has indicated it is finished
        if action is None:
            # If the policy exposes an 'episode_ended' flag, respect it
            if getattr(policy, 'episode_ended', False):
                print('Policy reported episode ended')
                break
            # Otherwise just wait for next cycle (e.g., teleop not enabled)
            # break

        # Execute valid action on robot
        if isinstance(action, dict):
            # Convert dict action to numpy array for the new environment API
            action_array = env.handler.dict_to_array(action)
            obs, reward, success, terminated, truncated, info = env.step(action_array)

            # Render and save images if enabled
            if env.handler.render_images:
                env.render()

            # Check if episode should end
            if terminated or truncated or success:
            # if truncated or success:
                print(f'Episode ended - Success: {success}, Terminated: {terminated}, Truncated: {truncated}')
                break

        # Episode ended
        elif action == 'end_episode':
            print('Episode ended')
            break

        # Ready for env reset
        elif action == 'reset_env':
            break

def main(args):
    # Create env
    if args.sim:
        from env.mujoco.mujoco_env import BlocksEnv, CabinetEnv, DrawerEnv
        # Use headless mode but enable rendering if saving images
        render_images = args.save_images
        env = BlocksEnv(render_images=render_images, show_viewer=False, max_episode_steps=100000, render_every_n_frames=100)
        if args.save_images:
            print("Simulation will run in headless mode with image saving enabled")
    else:
        from env.real.real_env import RealEnv
        env = RealEnv()

    # Create policy
    if args.goto_cabinet_handle:
        policy = GoToCabinetHandlePolicy()
    elif args.goto_cabinet_handle_right:
        policy = GoToCabinetHandlePolicyRight()
    elif args.motion_planner:
        policy = MotionPlannerPolicy()
    elif args.motion_planner_stack:
        policy = MotionPlannerPolicyStack()
    elif args.teleop:
        policy = TeleopPolicy()
    else:
        policy = RemotePolicy()

    run_episode(env, policy)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--teleop', action='store_true')
    parser.add_argument('--motion_planner', action='store_true')
    parser.add_argument('--motion_planner_stack', action='store_true', help='Stack cubes by picking the smallest x value cube and placing it on the largest x value cube')
    parser.add_argument('--goto-cabinet-handle', action='store_true', help='Move gripper to left cabinet handle pose')
    parser.add_argument('--goto-cabinet-handle-right', action='store_true', help='Move gripper to right cabinet handle pose')
    parser.add_argument('--save-images', action='store_true', help='Save rendered simulation images to disk (headless)')
    main(parser.parse_args())
