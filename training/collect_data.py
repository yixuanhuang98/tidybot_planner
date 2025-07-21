#!/usr/bin/env python3
"""
Data collection script for TidyBot table stacking environment.

This script collects demonstration data using the motion planner policy
and saves it in LeRobot format for training.
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from training.tidybot_env import TidybotEnv, TidybotPolicyWrapper


def collect_tidybot_dataset(
    repo_id: str = "your_username/tidybot_table_stack",
    num_episodes: int = 100,
    fps: int = 10,
    dataset_root: str = None,
    push_to_hub: bool = True,
    task_description: str = "Pick the cube with smallest x-coordinate and stack it on the cube with largest x-coordinate",
    max_episode_steps: int = 1000,
    show_viewer: bool = False,
    video_backend: str = "pyav",
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
    motion_planner: str = "motion_planner_table_stack",
):
    """
    Collect TidyBot dataset following LeRobot patterns.
    
    Args:
        repo_id: Dataset repository ID for HuggingFace Hub
        num_episodes: Number of episodes to collect
        fps: Frames per second for data collection
        dataset_root: Local root directory for dataset storage
        push_to_hub: Whether to push dataset to HuggingFace Hub
        task_description: Description of the task being performed
        max_episode_steps: Maximum steps per episode
        show_viewer: Whether to show MuJoCo viewer
        video_backend: Video backend for encoding
        image_writer_threads: Number of threads for image writing
        image_writer_processes: Number of processes for image writing
        motion_planner: Motion planner to use (determines task type)
    """
    
    # Determine task type from motion planner
    if 'cupboard' in motion_planner:
        task = 'cupboard'
        default_task_description = "Cupboard manipulation task"
    else:
        task = 'table_stack'
        default_task_description = "Pick the cube with smallest x-coordinate and stack it on the cube with largest x-coordinate"
    
    # Use custom task description if provided, otherwise use default
    if task_description == "Pick the cube with smallest x-coordinate and stack it on the cube with largest x-coordinate":
        task_description = default_task_description
    
    print(f"🤖 Starting TidyBot data collection")
    print(f"📊 Target episodes: {num_episodes}")
    print(f"🏗️  Motion planner: {motion_planner}")
    print(f"🎯 Task: {task} - {task_description}")
    print(f"📁 Dataset ID: {repo_id}")
    
    # 1. Create environment
    print("\n🏗️  Setting up environment...")
    try:
        env = TidybotEnv(
            task=task,
            show_viewer=show_viewer,
            render_images=True,
            max_episode_steps=max_episode_steps,
            render_every_n_frames=1
        )
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 2. Create policy wrapper
    print("🧠 Setting up motion planner policy...")
    policy_wrapper = TidybotPolicyWrapper(task=task)
    
    # 3. Define dataset features following LeRobot format
    print("📋 Defining dataset features...")
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (20,),
            "names": [
                "base_x", "base_y", "base_theta",
                "arm_x", "arm_y", "arm_z", 
                "arm_quat_w", "arm_quat_x", "arm_quat_y", "arm_quat_z",
                "gripper_pos",
                "cube1_x", "cube1_y", "cube1_z",
                "cube2_x", "cube2_y", "cube2_z", 
                "cube3_x", "cube3_y", "cube3_z"
            ]
        },
        "observation.images.base": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"]
        },
        "observation.images.wrist": {
            "dtype": "video", 
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"]
        },
        "observation.images.overview": {
            "dtype": "video",
            "shape": (3, 480, 640), 
            "names": ["channels", "height", "width"]
        },
        "action": {
            "dtype": "float32",
            "shape": (11,),
            "names": [
                "base_x", "base_y", "base_theta",
                "arm_x", "arm_y", "arm_z",
                "arm_quat_w", "arm_quat_x", "arm_quat_y", "arm_quat_z",
                "gripper_pos"
            ]
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    
    # 4. Create dataset
    print("💾 Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=dataset_root,
        use_videos=True,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
        features=features,
        video_backend=video_backend,
    )
    
    # 5. Collect episodes
    print(f"\n🎬 Starting data collection...")
    successful_episodes = 0
    total_attempts = 0
    
    while successful_episodes < num_episodes:
        total_attempts += 1
        print(f"\n📹 Recording episode {successful_episodes + 1}/{num_episodes} (attempt {total_attempts})")
        
        try:
            # Reset environment and policy
            obs, info = env.reset()
            policy_wrapper.reset()
            
            episode_start_time = time.time()
            episode_step = 0
            episode_reward = 0
            episode_success = False
            
            # Episode loop
            while True:
                loop_start_time = time.time()
                
                # Get action from policy
                action = policy_wrapper.select_action(obs)
                
                # Check if policy has finished
                if policy_wrapper.episode_ended:
                    print(f"   📋 Policy completed execution at step {episode_step}")
                    break
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Track episode statistics
                episode_step += 1
                episode_reward += reward
                episode_success = info.get('success', False)
                
                # Create frame for dataset
                frame = {
                    **obs,  # observation.state, observation.images.*
                    "action": action.astype(np.float32),
                    "next.reward": np.array([reward], dtype=np.float32),
                    "next.done": np.array([terminated or truncated], dtype=bool),
                }
                
                # Add frame to dataset
                dataset.add_frame(frame, task=task_description)
                
                # Update observation
                obs = next_obs
                
                # Check termination conditions
                if terminated or truncated:
                    print(f"   🏁 Episode terminated: terminated={terminated}, truncated={truncated}")
                    break
                
                # Maintain consistent timing
                if fps > 0:
                    dt_s = time.time() - loop_start_time
                    sleep_time = max(0, 1.0/fps - dt_s)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            # Episode statistics
            episode_duration = time.time() - episode_start_time
            print(f"   📊 Episode stats: {episode_step} steps, {episode_reward:.2f} reward, {episode_duration:.2f}s")
            print(f"   {'✅ SUCCESS' if episode_success else '❌ FAILED'}")
            
            # Save episode regardless of success (for learning from failures too)
            dataset.save_episode()
            successful_episodes += 1
            
            print(f"   💾 Episode {successful_episodes} saved")
            
        except Exception as e:
            print(f"   ❌ Error during episode collection: {e}")
            print(f"   🔄 Clearing episode buffer and continuing...")
            try:
                dataset.clear_episode_buffer()
            except:
                pass
            continue
    
    # 6. Finalize dataset
    print(f"\n🎉 Data collection complete! Collected {successful_episodes} episodes in {total_attempts} attempts")
    
    # Push to hub if requested
    if push_to_hub:
        print("📤 Pushing dataset to HuggingFace Hub...")
        try:
            dataset.push_to_hub()
            print("✅ Dataset successfully pushed to Hub!")
        except Exception as e:
            print(f"❌ Failed to push to Hub: {e}")
    
    # Cleanup
    env.close()
    
    print(f"\n📈 Dataset Summary:")
    print(f"   Repository ID: {repo_id}")
    print(f"   Episodes collected: {successful_episodes}")
    print(f"   Total frames: {dataset.num_frames}")
    print(f"   Dataset size: {dataset.num_frames} frames")
    print(f"   Local path: {dataset.root}")
    
    return dataset


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Collect TidyBot demonstration data")
    
    # Dataset configuration
    parser.add_argument("--repo-id", type=str, default="your_username/tidybot_table_stack",
                        help="Dataset repository ID")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for data collection")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Local root directory for dataset")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push dataset to HuggingFace Hub")
    parser.add_argument("--task-description", type=str, 
                        default="Pick the cube with smallest x-coordinate and stack it on the cube with largest x-coordinate",
                        help="Task description")
    
    # Environment configuration
    parser.add_argument("--max-episode-steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--show-viewer", action="store_true",
                        help="Show MuJoCo viewer")
    
    # Performance configuration
    parser.add_argument("--video-backend", type=str, default="pyav",
                        help="Video backend for encoding")
    parser.add_argument("--image-writer-threads", type=int, default=4,
                        help="Number of threads for image writing")
    parser.add_argument("--image-writer-processes", type=int, default=0,
                        help="Number of processes for image writing")
    parser.add_argument("--motion-planner", type=str, default="motion_planner_table_stack",
                        help="Motion planner to use (e.g., motion_planner_table_stack, motion_planner_cupboard)")
    
    args = parser.parse_args()
    
    # Collect dataset
    dataset = collect_tidybot_dataset(
        repo_id=args.repo_id,
        num_episodes=args.num_episodes,
        fps=args.fps,
        dataset_root=args.dataset_root,
        push_to_hub=args.push_to_hub,
        task_description=args.task_description,
        max_episode_steps=args.max_episode_steps,
        show_viewer=args.show_viewer,
        video_backend=args.video_backend,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        motion_planner=args.motion_planner,
    )
    
    print(f"✅ Data collection completed successfully!")


if __name__ == "__main__":
    main()