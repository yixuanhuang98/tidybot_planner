import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium.spaces import Box, Dict as DictSpace

from env.base_env import BaseEnv
from env.real.real_handler import RealHandler
from env.state import Action


class RealEnv(BaseEnv):
    """Real robot environment following standard API"""
    
    def __init__(self, use_cameras: bool = True, max_episode_steps: int = 1000):
        # Create handler
        self.handler = RealHandler(use_cameras=use_cameras)
        
        # Initialize parent class
        super().__init__(self.handler)
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_success = False

    def _setup_spaces(self):
        """Set up observation and action spaces for real robot"""
        # Action space: base pose (3), arm pose (3), arm quat (4), gripper (1)
        self._action_space = Box(
            low=np.array([-2.0, -2.0, -np.pi, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([2.0, 2.0, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: robot state (real robot doesn't have perfect object tracking)
        obs_dict = {
            'base_pose': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_quat': Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            'gripper_pos': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        }
        
        # Add camera observations if using cameras
        if self.handler.use_cameras:
            # Camera images - dimensions would depend on actual camera resolution
            obs_dict['base_image'] = Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
            obs_dict['wrist_image'] = Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
        
        self._observation_space = DictSpace(obs_dict)

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and return initial observation"""
        # Reset handler
        self.handler.set_states()  # Reset without action
        
        # Wait for robot to finish resetting
        import time
        time.sleep(2.0)  # Real robot needs time to reset
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_success = False
        
        # Get initial observation
        states = self.handler.get_states()
        observation = self.get_observation(states)
        info = self.handler.get_extra()
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, bool, Dict[str, Any]]:
        """Execute one environment step"""
        # Convert action array to Action object
        action_obj = self._array_to_action(action)
        
        # Execute action
        self.handler.set_states(action_obj)
        self.handler.step()
        
        # Get new state
        states = self.handler.get_states()
        observation = self.get_observation(states)
        
        # Calculate reward and termination
        reward = self.get_reward(states)
        success = self.get_success(states)
        terminated = self.get_termination(states)
        truncated = self.get_timeout(states)
        
        # Update episode tracking
        self.current_step += 1
        self.episode_reward += reward
        self.episode_success = self.episode_success or success
        
        # Get extra info
        info = self.handler.get_extra()
        info.update({
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'success': success
        })
        
        return observation, reward, success, terminated, truncated, info

    def _array_to_action(self, action: np.ndarray) -> Action:
        """Convert action array to Action object"""
        if len(action) != 11:
            raise ValueError(f"Expected action array of length 11, got {len(action)}")
        
        return Action(
            base_pose=action[:3],
            arm_pos=action[3:6],
            arm_quat=action[6:10],
            gripper_pos=action[10]
        )

    def get_observation(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Extract observation from states"""
        observation = {
            'base_pose': states['base_pose'].astype(np.float32),
            'arm_pos': states['arm_pos'].astype(np.float32),
            'arm_quat': states['arm_quat'].astype(np.float32),
            'gripper_pos': np.array([states['gripper_pos']], dtype=np.float32)
        }
        
        # Add camera images if available
        extra = self.handler.get_extra()
        if 'base_image' in extra:
            observation['base_image'] = extra['base_image']
        if 'wrist_image' in extra:
            observation['wrist_image'] = extra['wrist_image']
        
        return observation

    def get_reward(self, states: Dict[str, Any]) -> float:
        """Calculate reward from states - customizable for different tasks"""
        # Basic reward structure for real robot
        reward = 0.0
        
        # Small negative reward for each step to encourage efficiency
        reward -= 0.01
        
        # Safety rewards/penalties
        arm_pos = states['arm_pos']
        
        # Reward for keeping arm in safe workspace
        workspace_center = np.array([0.55, 0.0, 0.4])
        arm_distance = np.linalg.norm(arm_pos - workspace_center)
        if arm_distance < 0.5:
            reward += 0.05
        elif arm_distance > 1.0:
            reward -= 0.2  # Penalty for going too far
        
        # Penalty for arm going too low (safety)
        if arm_pos[2] < 0.1:
            reward -= 1.0
        
        return reward

    def get_success(self, states: Dict[str, Any]) -> bool:
        """Check if task was successful - override for specific tasks"""
        # Default success condition - can be overridden
        # For real robot, success often depends on task-specific conditions
        return False

    def get_termination(self, states: Dict[str, Any]) -> bool:
        """Check if episode should terminate for safety"""
        # Safety-based termination for real robot
        
        # Check if base moved too far from origin
        base_pose = states['base_pose']
        base_distance = np.linalg.norm(base_pose[:2])
        if base_distance > 2.0:  # More conservative for real robot
            print("Episode terminated: Base moved too far from origin")
            return True
        
        # Check if arm is in unsafe position
        arm_pos = states['arm_pos']
        if arm_pos[2] < 0.05:  # Very low arm position
            print("Episode terminated: Arm position unsafe (too low)")
            return True
        
        if np.linalg.norm(arm_pos) > 1.5:  # Arm extended too far
            print("Episode terminated: Arm extended too far")
            return True
        
        return False

    def get_timeout(self, states: Dict[str, Any]) -> bool:
        """Check if episode timed out"""
        return self.current_step >= self.max_episode_steps

    def render(self) -> Optional[Dict[str, Any]]:
        """Render the environment (return camera images)"""
        return self.handler.render()

    def close(self):
        """Close the environment"""
        self.handler.close()

    @property
    def observation_space(self):
        """Get observation space"""
        return self._observation_space

    @property
    def action_space(self):
        """Get action space"""
        return self._action_space