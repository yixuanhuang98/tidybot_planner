#!/usr/bin/env python3
"""
Training script for TidyBot policies using LeRobot framework.

This script trains various policies (Diffusion, ACT, VQ-BeT) on the collected
TidyBot demonstration data.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.utils.utils import init_logging

# Policy imports
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_policy_config(
    policy_type: str,
    input_features: Dict[str, Any],
    output_features: Dict[str, Any],
    **kwargs
) -> Any:
    """
    Create policy configuration based on policy type.
    
    Args:
        policy_type: Type of policy ("diffusion", "act", "vqbet")
        input_features: Input feature specifications
        output_features: Output feature specifications
        **kwargs: Additional configuration parameters
        
    Returns:
        Policy configuration object
    """
    if policy_type == "diffusion":
        return DiffusionConfig(
            # Observation history
            n_obs_steps=kwargs.get("n_obs_steps", 2),
            # Action chunking - predict 8 future actions
            horizon=kwargs.get("horizon", 16),
            n_action_steps=kwargs.get("n_action_steps", 8),
            # Diffusion parameters
            num_inference_steps=kwargs.get("num_inference_steps", 10),
            down_dims=kwargs.get("down_dims", (512, 1024, 2048)),
            kernel_size=kwargs.get("kernel_size", 5),
            n_groups=kwargs.get("n_groups", 8),
            # Vision encoder parameters
            vision_backbone=kwargs.get("vision_backbone", "resnet18"),
            crop_shape=kwargs.get("crop_shape", (84, 84)),
            crop_is_random=kwargs.get("crop_is_random", True),
            use_group_norm=kwargs.get("use_group_norm", True),
        )
    
    elif policy_type == "act":
        return ACTConfig(
            input_features=input_features,
            output_features=output_features,
            # Observation history
            observation_delta_indices=kwargs.get("observation_delta_indices", [-1, 0]),
            # Action chunking - predict 8 future actions
            action_delta_indices=kwargs.get("action_delta_indices", list(range(8))),
            # ACT parameters
            chunk_size=kwargs.get("chunk_size", 8),
            n_layers=kwargs.get("n_layers", 8),
            n_heads=kwargs.get("n_heads", 8),
            dim_feedforward=kwargs.get("dim_feedforward", 3200),
            hidden_dim=kwargs.get("hidden_dim", 512),
            dropout=kwargs.get("dropout", 0.1),
            # Vision encoder parameters
            vision_backbone=kwargs.get("vision_backbone", "resnet18"),
            crop_shape=kwargs.get("crop_shape", (84, 84)),
            crop_is_random=kwargs.get("crop_is_random", True),
            use_group_norm=kwargs.get("use_group_norm", True),
        )
    
    elif policy_type == "vqbet":
        return VQBeTConfig(
            input_features=input_features,
            output_features=output_features,
            # Observation history
            observation_delta_indices=kwargs.get("observation_delta_indices", [-1, 0]),
            # Action chunking - predict 8 future actions
            action_delta_indices=kwargs.get("action_delta_indices", list(range(8))),
            # VQ-BeT parameters
            n_layers=kwargs.get("n_layers", 8),
            n_heads=kwargs.get("n_heads", 8),
            dim_feedforward=kwargs.get("dim_feedforward", 3200),
            hidden_dim=kwargs.get("hidden_dim", 512),
            dropout=kwargs.get("dropout", 0.1),
            # Vision encoder parameters
            vision_backbone=kwargs.get("vision_backbone", "resnet18"),
            crop_shape=kwargs.get("crop_shape", (84, 84)),
            crop_is_random=kwargs.get("crop_is_random", True),
            use_group_norm=kwargs.get("use_group_norm", True),
        )
    
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def create_policy(
    policy_type: str,
    config: Any,
    dataset_stats: Dict[str, Any],
    device: str = "cuda"
) -> Any:
    """
    Create policy instance based on type and configuration.
    
    Args:
        policy_type: Type of policy ("diffusion", "act", "vqbet")
        config: Policy configuration
        dataset_stats: Dataset statistics for normalization
        device: Device for computation
        
    Returns:
        Policy instance
    """
    if policy_type == "diffusion":
        policy = DiffusionPolicy(config, dataset_stats=dataset_stats)
    elif policy_type == "act":
        policy = ACTPolicy(config, dataset_stats=dataset_stats)
    elif policy_type == "vqbet":
        policy = VQBeTPolicy(config, dataset_stats=dataset_stats)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    policy.train()
    policy.to(device)
    return policy


def train_tidybot_policy(
    dataset_repo_id: str,
    policy_type: str = "diffusion",
    output_dir: str = "outputs/train/tidybot",
    training_steps: int = 50000,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda",
    eval_freq: int = 5000,
    save_freq: int = 10000,
    log_freq: int = 100,
    num_workers: int = 4,
    dataset_root: str = None,
    wandb_project: str = None,
    wandb_entity: str = None,
    seed: int = 42,
    **policy_kwargs
) -> None:
    """
    Train TidyBot policy using LeRobot framework.
    
    Args:
        dataset_repo_id: Dataset repository ID
        policy_type: Type of policy to train ("diffusion", "act", "vqbet")
        output_dir: Directory to save training outputs
        training_steps: Number of training steps
        batch_size: Batch size for training
        lr: Learning rate
        device: Device for computation
        eval_freq: Frequency of evaluation
        save_freq: Frequency of saving checkpoints
        log_freq: Frequency of logging
        num_workers: Number of dataloader workers
        dataset_root: Local dataset root directory
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity name
        seed: Random seed
        **policy_kwargs: Additional policy configuration parameters
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"🤖 Starting TidyBot policy training")
    logger.info(f"📊 Dataset: {dataset_repo_id}")
    logger.info(f"🧠 Policy: {policy_type}")
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"🎯 Training steps: {training_steps}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset metadata to configure policy
    logger.info("📋 Loading dataset metadata...")
    dataset_metadata = LeRobotDataset(dataset_repo_id, root=dataset_root, video_backend="pyav")
    
    # Configure policy features
    logger.info("🔧 Configuring policy features...")
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # Separate input and output features
    input_features = {}
    output_features = {}
    
    for key, feature in features.items():
        if key == "action":
            output_features[key] = feature
        else:
            input_features[key] = feature
    
    logger.info(f"📥 Input features: {list(input_features.keys())}")
    logger.info(f"📤 Output features: {list(output_features.keys())}")
    
    # Create policy configuration
    logger.info(f"⚙️  Creating {policy_type} policy configuration...")
    config = create_policy_config(
        policy_type=policy_type,
        input_features=input_features,
        output_features=output_features,
        **policy_kwargs
    )
    
    # Set input and output features on the config
    config.input_features = input_features
    config.output_features = output_features
    
    # Create policy
    logger.info("🧠 Creating policy...")
    
    # Handle dataset stats
    if hasattr(dataset_metadata, 'stats'):
        dataset_stats = dataset_metadata.stats
    else:
        # Create minimal stats if not available
        logger.warning("⚠️  Dataset stats not available, creating minimal stats...")
        dataset_stats = {
            "observation.state": {
                "min": np.full(input_features["observation.state"].shape, -1.0),
                "max": np.full(input_features["observation.state"].shape, 1.0),
                "mean": np.full(input_features["observation.state"].shape, 0.0),
                "std": np.full(input_features["observation.state"].shape, 1.0),
            },
            "observation.images.base": {
                "min": np.full(input_features["observation.images.base"].shape, 0.0),
                "max": np.full(input_features["observation.images.base"].shape, 1.0),
                "mean": np.full(input_features["observation.images.base"].shape, 0.5),
                "std": np.full(input_features["observation.images.base"].shape, 0.25),
            },
            "observation.images.wrist": {
                "min": np.full(input_features["observation.images.wrist"].shape, 0.0),
                "max": np.full(input_features["observation.images.wrist"].shape, 1.0),
                "mean": np.full(input_features["observation.images.wrist"].shape, 0.5),
                "std": np.full(input_features["observation.images.wrist"].shape, 0.25),
            },
            "observation.images.overview": {
                "min": np.full(input_features["observation.images.overview"].shape, 0.0),
                "max": np.full(input_features["observation.images.overview"].shape, 1.0),
                "mean": np.full(input_features["observation.images.overview"].shape, 0.5),
                "std": np.full(input_features["observation.images.overview"].shape, 0.25),
            },
            "action": {
                "min": np.full(output_features["action"].shape, -1.0),
                "max": np.full(output_features["action"].shape, 1.0),
                "mean": np.full(output_features["action"].shape, 0.0),
                "std": np.full(output_features["action"].shape, 1.0),
            },
        }
    
    policy = create_policy(
        policy_type=policy_type,
        config=config,
        dataset_stats=dataset_stats,
        device=device
    )
    
    logger.info(f"📊 Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Define delta timestamps for multi-frame observations
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in config.observation_delta_indices],
        "observation.images.base": [i / dataset_metadata.fps for i in config.observation_delta_indices],
        "observation.images.wrist": [i / dataset_metadata.fps for i in config.observation_delta_indices],
        "observation.images.overview": [i / dataset_metadata.fps for i in config.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in config.action_delta_indices],
    }
    
    logger.info(f"⏱️  Delta timestamps: {delta_timestamps}")
    
    # Create dataset with delta timestamps
    logger.info("📊 Loading dataset with temporal configuration...")
    dataset = LeRobotDataset(
        dataset_repo_id, 
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav"
    )
    
    # Create dataloader
    logger.info("🔄 Creating dataloader...")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device != "cpu",
        drop_last=True
    )
    
    # Setup optimizer
    logger.info("📈 Setting up optimizer...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_steps, eta_min=lr * 0.1
    )
    
    # Setup Weights & Biases logging
    if wandb_project:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"tidybot_{policy_type}_{training_steps}steps",
                config={
                    "policy_type": policy_type,
                    "dataset": dataset_repo_id,
                    "training_steps": training_steps,
                    "batch_size": batch_size,
                    "lr": lr,
                    "device": device,
                    **policy_kwargs
                }
            )
            logger.info("📊 Weights & Biases logging enabled")
        except ImportError:
            logger.warning("⚠️  Weights & Biases not available, skipping logging")
            wandb = None
    else:
        wandb = None
    
    # Training loop
    logger.info("🚀 Starting training loop...")
    step = 0
    total_loss = 0.0
    
    while step < training_steps:
        for batch in dataloader:
            if step >= training_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            loss, _ = policy(batch)  # forward method returns (loss, None)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Update statistics
            total_loss += loss.item()
            step += 1
            
            # Logging
            if step % log_freq == 0:
                avg_loss = total_loss / log_freq
                current_lr = scheduler.get_last_lr()[0]
                
                logger.info(f"Step {step:6d}/{training_steps} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
                
                if wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "train/step": step
                    })
                
                total_loss = 0.0
            
            # Save checkpoint
            if step % save_freq == 0:
                checkpoint_dir = output_path / f"checkpoint_{step:06d}"
                checkpoint_dir.mkdir(exist_ok=True)
                
                logger.info(f"💾 Saving checkpoint at step {step}")
                policy.save_pretrained(checkpoint_dir)
                
                # Save training state
                torch.save({
                    'step': step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_dir / "training_state.pt")
    
    # Save final model
    logger.info("💾 Saving final model...")
    final_dir = output_path / "final"
    final_dir.mkdir(exist_ok=True)
    policy.save_pretrained(final_dir)
    
    # Close wandb run
    if wandb:
        wandb.finish()
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"📁 Final model saved to: {final_dir}")


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train TidyBot policy")
    
    # Dataset configuration
    parser.add_argument("--dataset-repo-id", type=str, required=True,
                        help="Dataset repository ID")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Local dataset root directory")
    
    # Policy configuration
    parser.add_argument("--policy-type", type=str, default="diffusion",
                        choices=["diffusion", "act", "vqbet"],
                        help="Type of policy to train")
    parser.add_argument("--output-dir", type=str, default="outputs/train/tidybot",
                        help="Output directory for training")
    
    # Training configuration
    parser.add_argument("--training-steps", type=int, default=50000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Logging configuration
    parser.add_argument("--log-freq", type=int, default=100,
                        help="Frequency of logging")
    parser.add_argument("--save-freq", type=int, default=10000,
                        help="Frequency of saving checkpoints")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Weights & Biases entity name")
    
    args = parser.parse_args()
    
    # Train policy
    train_tidybot_policy(
        dataset_repo_id=args.dataset_repo_id,
        policy_type=args.policy_type,
        output_dir=args.output_dir,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        dataset_root=args.dataset_root,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()