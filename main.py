# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
import threading
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from agent.teleop_policy import TeleopPolicy
from agent.remote_policy import RemotePolicy
from agent.motion_planner_policy import MotionPlannerPolicy
from agent.go_to_cabinet_handle_policy import GoToCabinetHandlePolicy
from agent.go_to_cabinet_handle_policy_right import GoToCabinetHandlePolicyRight
from agent.mmmp_policy import MMMPPolicy
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
        # if action is None:
        #     # If the policy exposes an 'episode_ended' flag, respect it
        #     if getattr(policy, 'episode_ended', False):
        #         print('Policy reported episode ended')
        #         break
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
            # if terminated or truncated or success:
            if truncated or success:
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
    # Initialize web renderer variable
    web_renderer = None
    web_thread = None
    
    # Create env
    if args.sim:
        from env.mujoco.mujoco_env import BlocksEnv, CabinetEnv, DrawerEnv
        
        # Use headless mode but enable rendering if saving images or using web renderer
        render_images = args.save_images or args.web_renderer
        env = BlocksEnv(render_images=render_images, show_viewer=False, max_episode_steps=100000, render_every_n_frames=100)
        
        if args.save_images:
            print("Simulation will run in headless mode with image saving enabled")
            
        # Initialize web renderer if requested
        if args.web_renderer:
            try:
                from env.mujoco.web_renderer import create_web_renderer
                print(f"🌐 Initializing web renderer on port {args.web_port}...")
                
                # Create web renderer
                web_renderer = create_web_renderer(
                    env.handler.model, 
                    env.handler.data, 
                    width=640, 
                    height=480, 
                    port=args.web_port
                )
                
                # Start web server in background thread
                web_thread = threading.Thread(
                    target=web_renderer.run,
                    kwargs={'host': '0.0.0.0', 'debug': False},
                    daemon=True
                )
                web_thread.start()
                
                print(f"✅ Web renderer started! Open your browser to: http://localhost:{args.web_port}")
                print("   The renderer will automatically stream the simulation in real-time")
                
            except ImportError as e:
                print(f"❌ Failed to import web renderer: {e}")
                print("   Make sure to install dependencies: pip install -r env/mujoco/requirements_web.txt")
                args.web_renderer = False
            except Exception as e:
                print(f"❌ Failed to initialize web renderer: {e}")
                args.web_renderer = False
    else:
        from env.real.real_env import RealEnv
        env = RealEnv()
        
        # Web renderer only works with simulation
        if args.web_renderer:
            print("⚠️  Web renderer is only available for simulation mode (--sim)")
            args.web_renderer = False

    # Create policy
    if args.goto_cabinet_handle:
        policy = GoToCabinetHandlePolicy()
    elif args.goto_cabinet_handle_right:
        policy = GoToCabinetHandlePolicyRight()
    elif args.motion_planner:
        policy = MotionPlannerPolicy()
    elif args.motion_planner_stack:
        policy = MotionPlannerPolicyStack()
    elif args.mmmp:
        policy = MMMPPolicy()
    elif args.teleop:
        policy = TeleopPolicy()
    else:
        policy = RemotePolicy()

    # Show web renderer information if enabled
    if args.web_renderer and args.sim:
        print("\n" + "="*60)
        print("🌐 WEB RENDERER ACTIVE")
        print(f"   URL: http://localhost:{args.web_port}")
        print("   • Real-time simulation visualization")
        print("   • Multiple camera views (Base, Wrist, Overview)")
        print("   • Live statistics and controls")
        print("   • The simulation will stream automatically!")
        print("="*60 + "\n")

    try:
        run_episode(env, policy)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        
        # Close web renderer if it was started
        if web_renderer:
            try:
                web_renderer.close()
                print("✅ Web renderer closed")
            except Exception as e:
                print(f"⚠️  Error closing web renderer: {e}")
        
        # Close environment
        env.close()
        print("✅ Environment closed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--teleop', action='store_true')
    parser.add_argument('--motion_planner', action='store_true')
    parser.add_argument('--motion_planner_stack', action='store_true', help='Stack cubes by picking the smallest x value cube and placing it on the largest x value cube')
    parser.add_argument('--mmmp', action='store_true', help='Move gripper to left cabinet handle pose')
    parser.add_argument('--goto-cabinet-handle', action='store_true', help='Move gripper to left cabinet handle pose')
    parser.add_argument('--goto-cabinet-handle-right', action='store_true', help='Move gripper to right cabinet handle pose')
    parser.add_argument('--save-images', action='store_true', help='Save rendered simulation images to disk (headless)')
    parser.add_argument('--web-renderer', action='store_true', help='Enable real-time web renderer for simulation visualization')
    parser.add_argument('--web-port', type=int, default=5000, help='Port for web renderer (default: 5000)')
    main(parser.parse_args())
