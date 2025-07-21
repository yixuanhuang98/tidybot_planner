#!/usr/bin/env python3
"""
Example usage of the TidyBot LeRobot integration.

This script demonstrates how to use the data collection, training, and evaluation
pipeline for different TidyBot tasks.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and print the description."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ Command completed successfully")
        return True

def main():
    """Main example workflow."""
    parser = argparse.ArgumentParser(description="TidyBot LeRobot Integration Example")
    parser.add_argument(
        '--task',
        type=str,
        default='table_stack',
        choices=['table_stack', 'cupboard'],
        help='Task to run (default: table_stack)'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default=None,
        help='Dataset repository ID (default: auto-generated based on task)'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1,
        help='Number of episodes to collect (default: 1)'
    )
    parser.add_argument(
        '--training-steps',
        type=int,
        default=5000,
        help='Number of training steps (default: 5000)'
    )
    parser.add_argument(
        '--motion-planner',
        type=str,
        default=None,
        help='Motion planner to use (default: auto-selected based on task)'
    )
    
    args = parser.parse_args()
    
    print("🤖 TidyBot LeRobot Integration Example")
    print("This script demonstrates the complete workflow from data collection to evaluation.")
    
    # Set task-specific defaults
    if args.task == 'table_stack':
        motion_planner = args.motion_planner or 'motion_planner_table_stack'
        repo_id = args.repo_id or "keplerccc/tidybot_table_stack_test"
    elif args.task == 'cupboard':
        motion_planner = args.motion_planner or 'motion_planner_cupboard'
        repo_id = args.repo_id or "keplerccc/tidybot_cupboard_test"
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    num_episodes = args.num_episodes
    training_steps = args.training_steps
    
    print(f"\n📋 Configuration:")
    print(f"   Task: {args.task}")
    print(f"   Motion Planner: {motion_planner}")
    print(f"   Dataset: {repo_id}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Training steps: {training_steps}")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "training").exists():
        print("❌ Error: Please run this script from the tidybot_planner directory")
        return
    
    # Step 1: Data Collection
    collect_cmd = f"""python training/collect_data.py \\
        --repo-id {repo_id} \\
        --num-episodes {num_episodes} \\
        --fps 10 \\
        --max-episode-steps 50000 \\
        --motion-planner {motion_planner}"""
    
    if not run_command(collect_cmd, "Data Collection"):
        return
    
    # Step 2: Policy Training (Diffusion)
    train_cmd = f"""python training/train_policy.py \\
        --dataset-repo-id {repo_id} \\
        --policy-type diffusion \\
        --output-dir outputs/train/tidybot_demo \\
        --training-steps {training_steps} \\
        --batch-size 4 \\
        --lr 1e-4 \\
        --log-freq 50 \\
        --save-freq 500"""
    
    if not run_command(train_cmd, "Policy Training"):
        return
    
    # Step 3: Policy Evaluation
    eval_cmd = f"""python training/evaluate_policy.py \\
        --policy-path outputs/train/tidybot_demo/final \\
        --dataset-repo-id {repo_id} \\
        --task {args.task} \\
        --num-episodes 1 \\
        --max-steps 3000 \\
        --save-videos \\
        --video-dir evaluation_videos \\
        --output-file evaluation_results.json"""
    
    if not run_command(eval_cmd, "Policy Evaluation"):
        return
    
    print("\n🎉 Example workflow completed successfully!")
    print("\nNext steps:")
    print("1. Increase num_episodes to 100+ for better policy performance")
    print("2. Increase training_steps to 50000+ for full training")
    print("3. Try different policy types (act, vqbet)")
    print("4. Experiment with different hyperparameters")
    print("5. Use --wandb-project for training monitoring")
    print("6. Try different tasks: --task cupboard or --task table_stack")

if __name__ == "__main__":
    main()