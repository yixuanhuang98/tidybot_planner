"""
Gymnasium wrapper for TidyBot environments.

This wrapper makes the existing MuJoCo environments compatible with LeRobot's
data collection and training pipeline. Supports multiple tasks including
table stacking and cupboard tasks.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional
import sys
import os

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.mujoco.mujoco_env import TableBlocksEnv, CupboardEnv
from agent.motion_planner_table_stack_policy import MotionPlannerTableStackPolicy
from agent.mp_policy import MotionPlannerPolicy_New


class TidybotEnv(gym.Env):
    """
    Gymnasium wrapper for TidyBot environments.
    
    This environment wraps the existing environments to be compatible with
    LeRobot's data collection and training pipeline.
    
    Supports multiple tasks:
    - table_stack: Pick the cube with smallest x-coordinate and stack it on the cube with largest x-coordinate
    - cupboard: Cupboard-related manipulation tasks
    """
    
    def __init__(self, 
                 task: str = 'table_stack',
                 show_viewer: bool = False,
                 render_images: bool = True,
                 max_episode_steps: int = 1000,
                 render_every_n_frames: int = 1,
                 **kwargs):
        """
        Initialize the TidyBot environment.
        
        Args:
            task: Task type ('table_stack' or 'cupboard')
            show_viewer: Whether to show the MuJoCo viewer
            render_images: Whether to render images from cameras
            max_episode_steps: Maximum steps per episode
            render_every_n_frames: Render frequency
        """
        super().__init__()
        self.task = task
        
        # Initialize the appropriate environment based on task
        if task == 'table_stack':
            self.env = TableBlocksEnv(
                show_viewer=show_viewer,
                render_images=render_images,
                max_episode_steps=max_episode_steps,
                render_every_n_frames=render_every_n_frames,
                save_images=True  # Don't save images to disk for data collection
            )
        elif task == 'cupboard':
            self.env = CupboardEnv(
                show_viewer=show_viewer,
                render_images=render_images,
                max_episode_steps=max_episode_steps,
                render_every_n_frames=render_every_n_frames,
                save_images=True  # Don't save images to disk for data collection
            )
        else:
            raise ValueError(f"Unknown task: {task}. Supported tasks: 'table_stack', 'cupboard'")
        
        # Define action space (11D: base_pose + arm_pos + arm_quat + gripper)
        self.action_space = spaces.Box(
            low=np.array([-2.0, -2.0, -np.pi, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([2.0, 2.0, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space following LeRobot pattern
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(20,),  # robot state (11) + cube positions (9)
                dtype=np.float32
            ),
            "observation.images.base": spaces.Box(
                low=0, 
                high=255, 
                shape=(3, 480, 640), 
                dtype=np.uint8
            ),
            "observation.images.wrist": spaces.Box(
                low=0, 
                high=255, 
                shape=(3, 480, 640), 
                dtype=np.uint8
            ),
            "observation.images.overview": spaces.Box(
                low=0, 
                high=255, 
                shape=(3, 480, 640), 
                dtype=np.uint8
            ),
        })
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = max_episode_steps
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment and return initial observation.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset the underlying environment
        obs, info = self.env.reset()
        
        # Reset episode tracking
        self.episode_step = 0
        
        # Format observation for LeRobot
        formatted_obs = self._format_observation(obs)
        
        return formatted_obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action array of shape (11,)
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Additional information
        """
        self.episode_step += 1
        
        # Execute action in the environment
        obs, reward, success, terminated, truncated, info = self.env.step(action)
        
        # Format observation for LeRobot
        formatted_obs = self._format_observation(obs)
        
        # Check for episode timeout
        if self.episode_step >= self.max_episode_steps:
            truncated = True
        
        # Update info with episode statistics
        info.update({
            'episode_step': self.episode_step,
            'success': success,
            'is_success': success,  # Common key for success tracking
        })
        
        return formatted_obs, reward, terminated, truncated, info
    
    def _format_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert observation to LeRobot format.
        
        Args:
            obs: Raw observation from TableBlocksEnv
            
        Returns:
            Formatted observation compatible with LeRobot
        """
        # Extract robot state components
        robot_state = np.concatenate([
            obs['base_pose'],      # 3D: base x, y, theta
            obs['arm_pos'],        # 3D: arm position
            obs['arm_quat'],       # 4D: arm quaternion
            obs['gripper_pos'],    # 1D: gripper position
        ])
        
        # Extract cube positions (3 cubes × 3 coordinates each)
        cube_state = np.concatenate([
            obs['cube1_pos'],      # 3D: cube 1 position
            obs['cube2_pos'],      # 3D: cube 2 position
            obs['cube3_pos'],      # 3D: cube 3 position
        ])
        
        # Combine all state information
        full_state = np.concatenate([robot_state, cube_state]).astype(np.float32)
        
        # Get camera images
        images = self.env.render()
        
        # Format images for LeRobot (convert HWC to CHW and ensure uint8)
        formatted_obs = {
            "observation.state": full_state,
        }
        
        # Add images if available
        if images is not None:
            for view_name, image in images.items():
                if image is not None:
                    # Convert HWC to CHW and ensure uint8
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8)
                    formatted_obs[f"observation.images.{view_name}"] = image.transpose(2, 0, 1)
                else:
                    # Create placeholder image if rendering failed
                    formatted_obs[f"observation.images.{view_name}"] = np.zeros((3, 480, 640), dtype=np.uint8)
        else:
            # Create placeholder images if no images available
            for view_name in ['base', 'wrist', 'overview']:
                formatted_obs[f"observation.images.{view_name}"] = np.zeros((3, 480, 640), dtype=np.uint8)
        
        return formatted_obs
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered image or None
        """
        if mode == 'rgb_array':
            images = self.env.render()
            if images and 'overview' in images:
                return images['overview']
        return None
    
    def close(self):
        """Close the environment and clean up resources."""
        self.env.close()
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional environment information."""
        return {
            'episode_step': self.episode_step,
            'max_episode_steps': self.max_episode_steps,
            'action_space_shape': self.action_space.shape,
            'observation_space_keys': list(self.observation_space.spaces.keys()),
        }


class TidybotPolicyWrapper:
    """
    Wrapper for the motion planner policy to work with the gym environment.
    """
    
    def __init__(self, task: str = 'table_stack'):
        """Initialize the policy wrapper.
        
        Args:
            task: Task type ('table_stack' or 'cupboard')
        """
        self.task = task
        
        if task == 'table_stack':
            self.policy = MotionPlannerTableStackPolicy()
        elif task == 'cupboard':
            self.policy = MotionPlannerPolicy_New(cupboard_mode=True, custom_grasp=True)
            self.policy.target_location = np.array([0.8, 0, 0.5])
        else:
            raise ValueError(f"Unknown task: {task}. Supported tasks: 'table_stack', 'cupboard'")
    
    @property
    def episode_ended(self):
        """Check if episode has ended from the policy."""
        return self.policy.episode_ended
    
    def reset(self):
        """Reset the policy for a new episode."""
        self.policy.reset()
    
    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            obs: Observation dictionary from TidybotEnv
            
        Returns:
            Action array of shape (11,)
        """
        # Convert LeRobot observation format back to original format
        original_obs = self._convert_observation(obs)
        
        # Get action from policy
        action_dict = self.policy.step(original_obs)
        
        if action_dict is None:
            # Policy has finished or failed
            # Return zero action
            return np.zeros(11, dtype=np.float32)
        
        # Convert action dict to array format
        action_array = np.concatenate([
            action_dict['base_pose'],      # 3D
            action_dict['arm_pos'],        # 3D
            action_dict['arm_quat'],       # 4D
            action_dict['gripper_pos'],    # 1D
        ]).astype(np.float32)
        
        return action_array
    
    def _convert_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert LeRobot observation format back to original format.
        
        Args:
            obs: LeRobot formatted observation
            
        Returns:
            Original observation format
        """
        state = obs['observation.state']
        
        # Extract components from state vector
        original_obs = {
            'base_pose': state[0:3],        # base x, y, theta
            'arm_pos': state[3:6],          # arm position
            'arm_quat': state[6:10],        # arm quaternion
            'gripper_pos': state[10:11],    # gripper position
            'cube1_pos': state[11:14],      # cube 1 position
            'cube2_pos': state[14:17],      # cube 2 position
            'cube3_pos': state[17:20],      # cube 3 position
        }
        
        return original_obs