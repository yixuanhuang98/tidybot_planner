#!/usr/bin/env python3
"""
Evaluation script for trained TidyBot policies.

This script evaluates trained policies on the TidyBot table stacking task
and computes success rates and other metrics.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
# from lerobot.policies.pretrained import PreTrainedPolicy  # Not needed - we load specific policy types
from lerobot.utils.utils import init_logging

from training.tidybot_env import TidybotEnv


def save_video_from_frames(frames: List[np.ndarray], video_path: str, fps: int = 30):
    """
    Save a list of frames as a video file.
    
    Args:
        frames: List of RGB frames (numpy arrays)
        video_path: Path to save the video file
        fps: Frames per second for the video
    """
    import cv2
    
    if len(frames) == 0:
        return
    
    # Get frame dimensions
    height, width, _ = frames[0].shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    # Release video writer
    video_writer.release()


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_policy(
    policy_path: str,
    device: str = "cuda",
    dataset_repo_id: str = None,
    dataset_root: str = None
) -> Any:
    """
    Load a trained policy from checkpoint.
    
    Args:
        policy_path: Path to the policy checkpoint
        device: Device for computation
        dataset_repo_id: Dataset repository ID for stats
        dataset_root: Local dataset root directory
        
    Returns:
        Loaded policy instance
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"📦 Loading policy from: {policy_path}")
    
    # Load policy configuration to determine type
    policy_path_obj = Path(policy_path)
    config_path = policy_path_obj / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Policy configuration not found: {config_path}")
    
    # Load the config to determine policy type
    import json
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    policy_type = config_dict.get('_policy_type', config_dict.get('type', 'diffusion'))
    
    # For now, just use the standard from_pretrained method with strict=False
    logger.info("📥 Loading policy (ignoring normalization shape mismatches)...")
    
    if policy_type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        
        # Create config manually, excluding the problematic 'type' field
        config_dict_clean = {k: v for k, v in config_dict.items() if k not in ['type', 'input_features', 'output_features']}
        
        # Convert normalization mapping to proper enum values
        if 'normalization_mapping' in config_dict_clean:
            from lerobot.configs.types import NormalizationMode
            norm_map = config_dict_clean['normalization_mapping']
            if isinstance(norm_map, dict):
                new_norm_map = {}
                for k, v in norm_map.items():
                    if v == "MEAN_STD":
                        new_norm_map[k] = NormalizationMode.MEAN_STD
                    elif v == "MIN_MAX":
                        new_norm_map[k] = NormalizationMode.MIN_MAX
                    else:
                        new_norm_map[k] = NormalizationMode.MEAN_STD
                config_dict_clean['normalization_mapping'] = new_norm_map
        
        config = DiffusionConfig(**config_dict_clean)
        
        # Set the input and output features
        config.input_features = {}
        config.output_features = {}
        
        # Create basic policy features from the config
        from lerobot.configs.types import PolicyFeature, FeatureType
        for key, feature_info in config_dict.get('input_features', {}).items():
            feature_type_str = feature_info.get('type', 'VISUAL')
            if feature_type_str == 'VISUAL':
                feature_type = FeatureType.VISUAL
            elif feature_type_str == 'STATE':
                feature_type = FeatureType.STATE
            elif feature_type_str == 'ACTION':
                feature_type = FeatureType.ACTION
            else:
                feature_type = FeatureType.VISUAL
                
            config.input_features[key] = PolicyFeature(
                type=feature_type,
                shape=tuple(feature_info.get('shape', []))
            )
        
        for key, feature_info in config_dict.get('output_features', {}).items():
            feature_type_str = feature_info.get('type', 'ACTION')
            if feature_type_str == 'ACTION':
                feature_type = FeatureType.ACTION
            else:
                feature_type = FeatureType.ACTION
                
            config.output_features[key] = PolicyFeature(
                type=feature_type,
                shape=tuple(feature_info.get('shape', []))
            )
        
        policy = DiffusionPolicy(config, dataset_stats=None)
        
        # Load weights manually with strict=False
        import safetensors.torch
        model_path = policy_path_obj / "model.safetensors"
        if model_path.exists():
            state_dict = safetensors.torch.load_file(model_path, device=device)
            
            # Filter out problematic normalization buffers that have shape mismatches
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if "normalize_inputs.buffer_observation_images" in key and ("mean" in key or "std" in key):
                    # Skip these problematic buffers - we'll initialize them properly
                    logger.debug(f"Skipping problematic buffer: {key}")
                    continue
                filtered_state_dict[key] = value
            
            missing, unexpected = policy.load_state_dict(filtered_state_dict, strict=False)
            if len(missing) > 0:
                logger.warning(f"Missing keys: {len(missing)} keys")
            if len(unexpected) > 0:
                logger.warning(f"Unexpected keys: {len(unexpected)} keys")
                
            # Initialize the missing normalization buffers with proper values
            logger.info("📊 Initializing missing normalization buffers...")
            import torch
            for key in missing:
                if "normalize_inputs.buffer_observation_images" in key:
                    if "mean" in key:
                        # Initialize mean to 0.5 for images (typical for normalized images)
                        policy.state_dict()[key].fill_(0.5)
                    elif "std" in key:
                        # Initialize std to 0.25 for images (typical for normalized images)
                        policy.state_dict()[key].fill_(0.25)
        else:
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
    elif policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(policy_path)
    elif policy_type == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
        policy = VQBeTPolicy.from_pretrained(policy_path)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    policy.eval()
    policy.to(device)
    
    logger.info("✅ Policy loaded successfully")
    return policy


def evaluate_episode(
    env: TidybotEnv,
    policy: Any,
    max_steps: int = 1000,
    render: bool = False,
    device: str = "cuda",
    save_video: bool = False,
    video_path: str = None
) -> Dict[str, Any]:
    """
    Evaluate a single episode.
    
    Args:
        env: Environment instance
        policy: Policy to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        device: Device for computation
        save_video: Whether to save video of the episode
        video_path: Path to save the video file
        
    Returns:
        Episode results dictionary
    """
    # Reset environment
    obs, info = env.reset()
    
    # Episode tracking
    episode_reward = 0.0
    episode_length = 0
    success = False
    episode_start_time = time.time()
    frames = []
    
    # Convert initial observation to tensor
    obs_tensor = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(device)
        else:
            obs_tensor[key] = torch.tensor(value).unsqueeze(0).to(device)
    
    # Capture initial frame if saving video
    if save_video:
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
    
    # Episode loop
    for step in range(max_steps):
        # Get action from policy
        with torch.no_grad():
            action_tensor = policy.select_action(obs_tensor)
        
        # Convert action to numpy
        if isinstance(action_tensor, torch.Tensor):
            action = action_tensor.cpu().numpy().flatten()
        else:
            action = np.array(action_tensor).flatten()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update episode statistics
        episode_reward += reward
        episode_length += 1
        success = info.get('success', False) or info.get('is_success', False)
        
        # Convert observation to tensor for next step
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(device)
            else:
                obs_tensor[key] = torch.tensor(value).unsqueeze(0).to(device)
        
        # Capture frame if saving video
        if save_video:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                frames.append(frame)
        
        # Render if requested
        if render:
            env.render()
        
        # Check termination
        if terminated or truncated:
            break
    
    # Save video if requested
    if save_video and video_path and len(frames) > 0:
        save_video_from_frames(frames, video_path)
    
    episode_duration = time.time() - episode_start_time
    
    return {
        'success': success,
        'reward': episode_reward,
        'length': episode_length,
        'duration': episode_duration,
        'terminated': terminated,
        'truncated': truncated,
        'final_info': info
    }


def evaluate_policy(
    policy_path: str,
    num_episodes: int = 100,
    device: str = "cuda",
    dataset_repo_id: str = None,
    dataset_root: str = None,
    max_steps: int = 1000,
    render: bool = False,
    show_viewer: bool = False,
    seed: int = 42,
    save_videos: bool = False,
    video_dir: str = "evaluation_videos"
) -> Dict[str, Any]:
    """
    Evaluate a trained policy on multiple episodes.
    
    Args:
        policy_path: Path to the policy checkpoint
        num_episodes: Number of episodes to evaluate
        device: Device for computation
        dataset_repo_id: Dataset repository ID for stats
        dataset_root: Local dataset root directory
        max_steps: Maximum steps per episode
        render: Whether to render during evaluation
        show_viewer: Whether to show MuJoCo viewer
        seed: Random seed for reproducibility
        save_videos: Whether to save evaluation videos
        video_dir: Directory to save videos
        
    Returns:
        Evaluation results dictionary
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎯 Starting policy evaluation")
    logger.info(f"📦 Policy: {policy_path}")
    logger.info(f"🎮 Episodes: {num_episodes}")
    logger.info(f"🖥️  Device: {device}")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load policy
    policy = load_policy(
        policy_path=policy_path,
        device=device,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root
    )
    
    # Create environment
    logger.info("🏗️  Creating evaluation environment...")
    env = TidybotEnv(
        show_viewer=show_viewer,
        render_images=render or save_videos,
        max_episode_steps=max_steps,
        render_every_n_frames=1
    )
    
    # Evaluation results
    results = []
    success_count = 0
    total_reward = 0.0
    total_steps = 0
    total_duration = 0.0
    
    # Create video directory if needed
    if save_videos:
        video_path = Path(video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation episodes
    logger.info("🚀 Starting evaluation episodes...")
    for episode in range(num_episodes):
        logger.info(f"📹 Episode {episode + 1}/{num_episodes}")
        
        # Generate video path if saving videos
        episode_video_path = None
        if save_videos:
            episode_video_path = video_path / f"episode_{episode+1:03d}.mp4"
        
        # Evaluate episode
        episode_result = evaluate_episode(
            env=env,
            policy=policy,
            max_steps=max_steps,
            render=render,
            device=device,
            save_video=save_videos,
            video_path=str(episode_video_path) if episode_video_path else None
        )
        
        # Log episode results
        success = episode_result['success']
        reward = episode_result['reward']
        length = episode_result['length']
        duration = episode_result['duration']
        
        logger.info(f"   {'✅' if success else '❌'} Success: {success}")
        logger.info(f"   📊 Reward: {reward:.2f}")
        logger.info(f"   📏 Length: {length} steps")
        logger.info(f"   ⏱️  Duration: {duration:.2f}s")
        if save_videos and episode_video_path:
            logger.info(f"   🎬 Video saved: {episode_video_path}")
        
        # Update statistics
        results.append(episode_result)
        if success:
            success_count += 1
        total_reward += reward
        total_steps += length
        total_duration += duration
    
    # Compute final statistics
    success_rate = success_count / num_episodes
    avg_reward = total_reward / num_episodes
    avg_steps = total_steps / num_episodes
    avg_duration = total_duration / num_episodes
    
    # Additional statistics
    rewards = [r['reward'] for r in results]
    lengths = [r['length'] for r in results]
    durations = [r['duration'] for r in results]
    
    evaluation_results = {
        'success_rate': success_rate,
        'success_count': success_count,
        'total_episodes': num_episodes,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_duration': avg_duration,
        'reward_std': np.std(rewards),
        'length_std': np.std(lengths),
        'duration_std': np.std(durations),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'episode_results': results
    }
    
    # Log final results
    logger.info("🎉 Evaluation completed!")
    logger.info(f"📊 Final Results:")
    logger.info(f"   Success Rate: {success_rate:.2%} ({success_count}/{num_episodes})")
    logger.info(f"   Average Reward: {avg_reward:.2f} ± {np.std(rewards):.2f}")
    logger.info(f"   Average Steps: {avg_steps:.1f} ± {np.std(lengths):.1f}")
    logger.info(f"   Average Duration: {avg_duration:.2f}s ± {np.std(durations):.2f}s")
    
    # Log video information
    if save_videos:
        logger.info(f"🎬 Videos saved to: {video_path}")
        logger.info(f"📹 Total videos: {num_episodes}")
    
    # Cleanup
    env.close()
    
    return evaluation_results


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained TidyBot policy")
    
    # Policy configuration
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the policy checkpoint")
    parser.add_argument("--dataset-repo-id", type=str, default=None,
                        help="Dataset repository ID for stats")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Local dataset root directory")
    
    # Evaluation configuration
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Visualization configuration
    parser.add_argument("--render", action="store_true",
                        help="Render during evaluation")
    parser.add_argument("--show-viewer", action="store_true",
                        help="Show MuJoCo viewer")
    parser.add_argument("--save-videos", action="store_true",
                        help="Save evaluation videos")
    parser.add_argument("--video-dir", type=str, default="evaluation_videos",
                        help="Directory to save videos")
    
    # Output configuration
    parser.add_argument("--output-file", type=str, default="evaluation_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Run evaluation
    results = evaluate_policy(
        policy_path=args.policy_path,
        num_episodes=args.num_episodes,
        device=args.device,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        max_steps=args.max_steps,
        render=args.render,
        show_viewer=args.show_viewer,
        seed=args.seed,
        save_videos=args.save_videos,
        video_dir=args.video_dir
    )
    
    # Save results to file
    if args.output_file:
        import json
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to python types for JSON serialization
        results_json = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_json[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                results_json[key] = value.item()
            elif key == 'episode_results':
                # Skip detailed episode results for JSON
                continue
            else:
                results_json[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"📄 Results saved to: {output_path}")
    
    print("✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()