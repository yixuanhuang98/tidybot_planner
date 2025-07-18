#!/usr/bin/env python3
"""
Example usage of the TidyBot LeRobot integration.

This script demonstrates how to use the data collection, training, and evaluation
pipeline for the TidyBot table stacking task.
"""

import os
import sys
import subprocess
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
    print("🤖 TidyBot LeRobot Integration Example")
    print("This script demonstrates the complete workflow from data collection to evaluation.")
    
    # Configuration
    repo_id = "keplerccc/tidybot_table_stack_test"
    num_episodes = 1  # Very small number for testing
    training_steps = 5000  # Very small number for testing
    
    print(f"\n📋 Configuration:")
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
        --max-episode-steps 5000"""
    
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
        --num-episodes 1 \\
        --max-steps 50 \\
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

if __name__ == "__main__":
    main()