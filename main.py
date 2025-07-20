# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
import json
import os
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
from policies import TeleopPolicy, RemotePolicy, MotionPlannerPolicy
from policies import MotionPlannerPolicyStackWrapper
from policies import MotionPlannerPolicyStackThreeWrapper
import numpy as np

def load_vlm_target_locations(vlm_json_path):
    """
    Load target locations from VLM JSON output file.
    
    Args:
        vlm_json_path (str): Path to the VLM JSON output file
        
    Returns:
        list: List of numpy arrays representing target locations [x, y, z]
    """
    try:
        with open(vlm_json_path, 'r') as f:
            vlm_data = json.load(f)
        
        target_locations = []
        x_offset = 0.75
        y_offset = -0.15
        z_offset = 0.27 # 7 cm for the placement. 
        for cup_data in vlm_data:
            cup_id = cup_data['cup_id']
            position = cup_data['position']
            # Convert to numpy array and ensure proper format
            target_loc = np.array([position['x'] + x_offset, position['y'] + y_offset, position['z'] + z_offset])
            target_locations.append(target_loc)
            print(f"Loaded cup {cup_id} target location: {target_loc}")
        
        print(f"Successfully loaded {len(target_locations)} target locations from VLM output")
        return target_locations
        
    except FileNotFoundError:
        print(f"Error: VLM JSON file not found at {vlm_json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {vlm_json_path}: {e}")
        return None
    except KeyError as e:
        print(f"Error: Missing required key in VLM JSON: {e}")
        return None
    except Exception as e:
        print(f"Error loading VLM target locations: {e}")
        return None

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
            if args.custom_grasp or args.custom_grasp_three:
                env = MujocoEnv(cupboard_scene=True, custom_grasp=True)
            else:
                env = MujocoEnv(cupboard_scene=True)
        elif args.cabinet_scene:
            env = MujocoEnv(cabinet_scene=True)
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
        if args.cupboard_scene:
            from policies import MotionPlannerPolicyMPCupboardWrapper
            policy = MotionPlannerPolicyMPCupboardWrapper(custom_grasp=args.custom_grasp)
        elif args.cabinet_scene:
            from policies import MotionPlannerPolicyMPCabinetWrapper
            policy = MotionPlannerPolicyMPCabinetWrapper(custom_grasp=args.custom_grasp)
        else:
            from policies import MotionPlannerPolicyMPWrapper
            policy = MotionPlannerPolicyMPWrapper(custom_grasp=args.custom_grasp)
    elif args.mp_policy_three:
        from policies import MotionPlannerPolicyMPThreeWrapper
        policy = MotionPlannerPolicyMPThreeWrapper(custom_grasp=args.custom_grasp)
    elif args.custom_grasp:
        from policies import MotionPlannerPolicyCustomGraspWrapper
        policy = MotionPlannerPolicyCustomGraspWrapper()
    elif args.custom_grasp_three:
        from policies import MotionPlannerPolicyCustomGraspThreeWrapper
        policy = MotionPlannerPolicyCustomGraspThreeWrapper()
    elif args.mp_policy_n_cupboard:
        from policies import MotionPlannerPolicyMPNCupboardWrapper
        
        # Load target locations from VLM if specified, otherwise use default
        if args.vlm:
            target_locations = load_vlm_target_locations(args.vlm_json_path)
            print('target_locations', target_locations)
            # time.sleep(15)
            # if target_locations is None:
            #     print("Error: Failed to load VLM target locations, using default locations")
            #     target_locations = [
            #         np.array([0.8, 0.08, 0.38]),   # Center position
            #         np.array([0.8, -0.08, 0.38]),  # Left position  
            #         np.array([0.73, 0, 0.38]),     # Right position
            #         np.array([0.8, 0.16, 0.38]),   # Far right position
            #         np.array([0.8, -0.16, 0.38])   # Far left position
            #     ]
        else:
            # Default target locations - can be customized
            target_locations = [
                np.array([0, 0, 0.38]),   # Center position
                np.array([0.8, -0.08, 0.38]),  # Left position  
                np.array([0.73, 0, 0.38]),     # Right position
                np.array([0.8, 0.16, 0.38]),   # Far right position
                np.array([0.8, -0.16, 0.38])   # Far left position
            ]
        
        # Custom grasp parameters (optional)
        # grasp_params = {
        #     'GRASP_SUCCESS_THRESHOLD': 0.75,
        #     'PICK_LOWER_DIST': 0.09,
        #     'PICK_LIFT_DIST': 0.18
        # }
        policy = MotionPlannerPolicyMPNCupboardWrapper(
            target_locations=target_locations,
            custom_grasp=args.custom_grasp
        )
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
    parser.add_argument('--mp_policy_cupboard', action='store_true', help='Enable new motion planner policy (cupboard mode) from agent/mp_policy.py')
    parser.add_argument('--mp_policy_three', action='store_true', help='Enable new motion planner policy (three sequential placements) from agent/mp_policy.py')
    parser.add_argument('--mp_policy_n_cupboard', action='store_true', help='Enable new motion planner policy (N sequential pick-place actions in cupboard) from agent/mp_policy.py')
    parser.add_argument('--vlm', action='store_true', help='Use VLM-generated target locations for mp_policy_n_cupboard')
    parser.add_argument('--vlm-json-path', type=str, default='VLM/vlm_target_locations.json', help='Path to VLM JSON output file containing target locations (default: VLM/vlm_target_locations.json)')
    parser.add_argument('--custom_grasp', action='store_true', help='Enable custom grasping mode for experimentation')
    parser.add_argument('--custom_grasp_three', action='store_true', help='Enable custom grasping mode for three sequential pick-place actions in cupboard environment')
    parser.add_argument('--stack_policy', action='store_true', help='Enable stacking policy (stack cubes)')
    parser.add_argument('--stack_policy_three', action='store_true', help='Enable stacking policy (stack three cubes) - works with both ground and table scenes')
    parser.add_argument('--table_scene', action='store_true', help='Use table scene with three cubes')
    parser.add_argument('--drawer_scene', action='store_true', help='Use drawer scene with three cubes')
    parser.add_argument('--cupboard_scene', action='store_true', help='Use cupboard scene with three cubes')
    parser.add_argument('--cabinet_scene', action='store_true', help='Use cabinet scene with three cubes')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output-dir', default='data/demos')
    main(parser.parse_args())
