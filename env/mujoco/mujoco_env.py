import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace

from env.base_env import BaseEnv
from env.mujoco.mujoco_handler import MujocoHandler
from env.state import EnvironmentState, Observation, Action


class AbstractMujocoEnv(BaseEnv, ABC):
    """Abstract MuJoCo environment that can be inherited by different scene types"""
    
    def __init__(self, mjcf_path: str, show_viewer: bool = False, render_images: bool = False,
                 max_episode_steps: int = 1000, render_every_n_frames: int = 1):
        
        # Create handler
        self.handler = MujocoHandler(mjcf_path=mjcf_path, 
                                   show_viewer=show_viewer, 
                                   render_images=render_images,
                                   render_every_n_frames=render_every_n_frames)
        
        # Initialize parent class
        super().__init__(self.handler)
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_success = False
        
        # Define observation and action spaces (to be set by subclasses)
        self._observation_space = None
        self._action_space = None
        
        # Setup spaces for this specific scene
        self._setup_spaces()

    @abstractmethod
    def _setup_spaces(self):
        """Set up observation and action spaces - must be implemented by subclasses"""
        pass

    @abstractmethod
    def _get_scene_specific_observation(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Get scene-specific observations - must be implemented by subclasses"""
        pass

    @abstractmethod
    def _get_scene_specific_reward(self, states: Dict[str, Any]) -> float:
        """Calculate scene-specific reward - must be implemented by subclasses"""
        pass

    @abstractmethod
    def _get_scene_specific_success(self, states: Dict[str, Any]) -> bool:
        """Check scene-specific success condition - must be implemented by subclasses"""
        pass

    @abstractmethod
    def _get_scene_specific_termination(self, states: Dict[str, Any]) -> bool:
        """Check scene-specific termination condition - must be implemented by subclasses"""
        pass

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and return initial observation"""
        # Reset handler
        self.handler.set_states()  # Reset without action
        
        # Wait for state to be updated
        import time
        time.sleep(0.1)
        
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
        observation = {}
        
        # Base robot observations (common to all scenes)
        observation.update(self._get_base_robot_observation(states))
        
        # Scene-specific observations
        observation.update(self._get_scene_specific_observation(states))
        
        return observation

    def _get_base_robot_observation(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Get base robot observations common to all scenes"""
        return {
            'base_pose': states['base_pose'].astype(np.float32),
            'arm_pos': states['arm_pos'].astype(np.float32),
            'arm_quat': states['arm_quat'].astype(np.float32),
            'gripper_pos': np.array([states['gripper_pos']], dtype=np.float32)
        }

    def get_reward(self, states: Dict[str, Any]) -> float:
        """Calculate reward from states"""
        # Base reward (common to all scenes)
        reward = self._get_base_reward(states)
        
        # Scene-specific reward
        reward += self._get_scene_specific_reward(states)
        
        return reward

    def _get_base_reward(self, states: Dict[str, Any]) -> float:
        """Get base reward common to all scenes"""
        reward = 0.0
        
        # Small negative reward for each step to encourage efficiency
        reward -= 0.01
        
        # Penalty for excessive base movement
        base_pose = states['base_pose']
        base_movement = np.linalg.norm(base_pose[:2])
        if base_movement > 1.0:
            reward -= 0.1
        
        return reward

    def get_success(self, states: Dict[str, Any]) -> bool:
        """Check if task was successful"""
        return self._get_scene_specific_success(states)

    def get_termination(self, states: Dict[str, Any]) -> bool:
        """Check if episode should terminate"""

        # Scene-specific termination
        return self._get_scene_specific_termination(states)

    def _get_base_termination(self, states: Dict[str, Any]) -> bool:
        """Check base termination conditions common to all scenes"""
        # Terminate if robot goes too far from origin
        base_pose = states['base_pose']
        base_distance = np.linalg.norm(base_pose[:2])
        
        if base_distance > 3.0:
            return True
        
        # Terminate if arm goes to unsafe position
        arm_pos = states['arm_pos']
        if arm_pos[2] < -0.1:  # Below table
            return True
        
        return False

    def get_timeout(self, states: Dict[str, Any]) -> bool:
        """Check if episode timed out"""
        return self.current_step >= self.max_episode_steps

    def render(self) -> Optional[Dict[str, Any]]:
        """Render the environment"""
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


class BlocksEnv(AbstractMujocoEnv):
    """Environment for block manipulation tasks"""
    
    def __init__(self, show_viewer: bool = False, render_images: bool = False,
                 max_episode_steps: int = 1000, render_every_n_frames: int = 1):
        super().__init__(
            mjcf_path="env/assets/stanford_tidybot/blocks_scene.xml",
            show_viewer=show_viewer,
            render_images=render_images,
            max_episode_steps=max_episode_steps,
            render_every_n_frames=render_every_n_frames
        )


class TableBlocksEnv(AbstractMujocoEnv):
    """Environment for table-based block manipulation tasks"""
    
    def __init__(self, show_viewer: bool = False, render_images: bool = False,
                 max_episode_steps: int = 1000, render_every_n_frames: int = 1):
        super().__init__(
            mjcf_path="env/assets/stanford_tidybot/blocks_table_scene.xml",
            show_viewer=show_viewer,
            render_images=render_images,
            max_episode_steps=max_episode_steps,
            render_every_n_frames=render_every_n_frames
        )

    def _setup_spaces(self):
        """Set up observation and action spaces for table blocks environment"""
        # Action space: base pose (3), arm pose (3), arm quat (4), gripper (1)
        self._action_space = Box(
            low=np.array([-2.0, -2.0, -np.pi, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([2.0, 2.0, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: robot state + cube states (same as BlocksEnv)
        obs_dict = {
            'base_pose': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_quat': Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            'gripper_pos': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        }
        
        # Add cube observations
        for obj_name in ['cube1', 'cube2', 'cube3']:
            obs_dict[f'{obj_name}_pos'] = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            obs_dict[f'{obj_name}_quat'] = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self._observation_space = DictSpace(obs_dict)

    def _get_scene_specific_observation(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Get cube observations for table environment"""
        observation = {}
        
        # Cube observations (same logic as BlocksEnv)
        for obj_name in ['cube1', 'cube2', 'cube3']:
            if f'{obj_name}_pos' in states:
                observation[f'{obj_name}_pos'] = states[f'{obj_name}_pos'].astype(np.float32)
            else:
                observation[f'{obj_name}_pos'] = np.zeros(3, dtype=np.float32)
            
            if f'{obj_name}_quat' in states:
                observation[f'{obj_name}_quat'] = states[f'{obj_name}_quat'].astype(np.float32)
            else:
                observation[f'{obj_name}_quat'] = np.array([0., 0., 0., 1.], dtype=np.float32)
        
        return observation

    def _get_scene_specific_reward(self, states: Dict[str, Any]) -> float:
        """Calculate reward for table block manipulation"""
        reward = 0.0
        
        # Reward for keeping arm in reasonable workspace above table
        arm_pos = states['arm_pos']
        table_workspace_center = np.array([0.55, 0.0, 0.5])  # Adjusted for table height
        arm_distance = np.linalg.norm(arm_pos - table_workspace_center)
        if arm_distance < 0.6:  # Slightly larger workspace for table
            reward += 0.1
        
        # Reward for cube stacking on table
        cube1_pos = states.get('cube1_pos', np.zeros(3))
        cube2_pos = states.get('cube2_pos', np.zeros(3))
        cube3_pos = states.get('cube3_pos', np.zeros(3))
        
        # Check if any cubes are stacked (adjusted heights for table environment)
        cube_pairs = [(cube1_pos, cube2_pos), (cube1_pos, cube3_pos), (cube2_pos, cube3_pos)]
        for pos1, pos2 in cube_pairs:
            if np.linalg.norm(pos1[:2] - pos2[:2]) < 0.05:  # Horizontal alignment
                height_diff = abs(pos2[2] - pos1[2])
                if 0.03 < height_diff < 0.08:  # Proper stacking height for table cubes
                    reward += 2.0  # Higher reward for successful table stacking
        
        return reward

    def _get_scene_specific_success(self, states: Dict[str, Any]) -> bool:
        """Check if blocks are successfully stacked on table"""
        cube1_pos = states.get('cube1_pos', np.zeros(3))
        cube2_pos = states.get('cube2_pos', np.zeros(3))
        cube3_pos = states.get('cube3_pos', np.zeros(3))
        
        # Check if any cubes are stacked (adjusted for table environment)
        cube_pairs = [(cube1_pos, cube2_pos), (cube1_pos, cube3_pos), (cube2_pos, cube3_pos)]
        for pos1, pos2 in cube_pairs:
            horizontal_distance = np.linalg.norm(pos1[:2] - pos2[:2])
            height_diff = abs(pos2[2] - pos1[2])
            
            # Success if cubes are stacked on table
            if horizontal_distance < 0.05 and 0.03 < height_diff < 0.08:
                return True
        
        return False

    def _get_scene_specific_termination(self, states: Dict[str, Any]) -> bool:
        """Check if blocks have fallen off table"""
        table_surface_height = 0.42  # Table surface height
        
        for obj_name in ['cube1', 'cube2', 'cube3']:
            pos = states.get(f'{obj_name}_pos', np.zeros(3))
            # Terminate if cube falls below table surface
            if pos[2] < table_surface_height - 0.05:  # 5cm below table surface
                print(f"Terminating: {obj_name} fell off table at height {pos[2]:.3f}")
                return True
            
            # Terminate if cube is too far from table horizontally
            table_center = np.array([0.6, 0.0])  # Table center in X,Y
            cube_distance_from_table = np.linalg.norm(pos[:2] - table_center)
            if cube_distance_from_table > 0.5:  # More than 50cm from table center
                print(f"Terminating: {obj_name} too far from table at distance {cube_distance_from_table:.3f}")
                
                return True
            
            return False


class CabinetEnv(AbstractMujocoEnv):
    """Environment for cabinet manipulation tasks"""
    
    def __init__(self, show_viewer: bool = False, render_images: bool = False,
                 max_episode_steps: int = 1000, render_every_n_frames: int = 1):
        super().__init__(
            mjcf_path="env/assets/stanford_tidybot/scene.xml",
            show_viewer=show_viewer,
            render_images=render_images,
            max_episode_steps=max_episode_steps,
            render_every_n_frames=render_every_n_frames
        )

    def _setup_spaces(self):
        """Set up observation and action spaces for cabinet environment"""
        # Action space: base pose (3), arm pose (3), arm quat (4), gripper (1)
        self._action_space = Box(
            low=np.array([-2.0, -2.0, -np.pi, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([2.0, 2.0, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: robot state + door positions
        obs_dict = {
            'base_pose': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_quat': Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            'gripper_pos': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'leftdoor_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'rightdoor_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        }
        
        self._observation_space = DictSpace(obs_dict)

    def _get_scene_specific_observation(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Get cabinet door observations"""
        return {
            'leftdoor_pos': states.get('leftdoor_pos', np.zeros(3)).astype(np.float32),
            'rightdoor_pos': states.get('rightdoor_pos', np.zeros(3)).astype(np.float32),
        }

    def _get_scene_specific_reward(self, states: Dict[str, Any]) -> float:
        """Calculate reward for cabinet manipulation"""
        reward = 0.0
        
        # Reward for getting close to door handles
        arm_pos = states['arm_pos']
        leftdoor_pos = states.get('leftdoor_pos', np.zeros(3))
        rightdoor_pos = states.get('rightdoor_pos', np.zeros(3))
        
        # Distance to nearest door handle
        left_distance = np.linalg.norm(arm_pos - leftdoor_pos)
        right_distance = np.linalg.norm(arm_pos - rightdoor_pos)
        min_distance = min(left_distance, right_distance)
        
        if min_distance < 0.2:
            reward += 0.5
        if min_distance < 0.1:
            reward += 1.0
        
        return reward

    def _get_scene_specific_success(self, states: Dict[str, Any]) -> bool:
        """Check if cabinet door is successfully opened"""
        # Success if gripper is close to handle and gripper is closed
        arm_pos = states['arm_pos']
        gripper_pos = states['gripper_pos']
        
        leftdoor_pos = states.get('leftdoor_pos', np.zeros(3))
        rightdoor_pos = states.get('rightdoor_pos', np.zeros(3))
        
        left_distance = np.linalg.norm(arm_pos - leftdoor_pos)
        right_distance = np.linalg.norm(arm_pos - rightdoor_pos)
        min_distance = min(left_distance, right_distance)
        
        return min_distance < 0.05 and gripper_pos < 0.1

    def _get_scene_specific_termination(self, states: Dict[str, Any]) -> bool:
        """Check cabinet-specific termination conditions"""
        # No additional termination conditions for cabinet
        return False


class DrawerEnv(AbstractMujocoEnv):
    """Environment for drawer manipulation tasks"""
    
    def __init__(self, show_viewer: bool = False, render_images: bool = False,
                 max_episode_steps: int = 1000, render_every_n_frames: int = 1):
        super().__init__(
            mjcf_path="env/assets/stanford_tidybot/drawer_scene.xml",  # Assume different scene
            show_viewer=show_viewer,
            render_images=render_images,
            max_episode_steps=max_episode_steps,
            render_every_n_frames=render_every_n_frames
        )

    def _setup_spaces(self):
        """Set up observation and action spaces for drawer environment"""
        # Action space: base pose (3), arm pose (3), arm quat (4), gripper (1)
        self._action_space = Box(
            low=np.array([-2.0, -2.0, -np.pi, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([2.0, 2.0, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: robot state + drawer handle position
        obs_dict = {
            'base_pose': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_quat': Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            'gripper_pos': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'drawer_handle_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        }
        
        self._observation_space = DictSpace(obs_dict)

    def _get_scene_specific_observation(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Get drawer handle observations"""
        return {
            'drawer_handle_pos': states.get('drawer_handle_pos', np.zeros(3)).astype(np.float32),
        }

    def _get_scene_specific_reward(self, states: Dict[str, Any]) -> float:
        """Calculate reward for drawer manipulation"""
        reward = 0.0
        
        # Reward for getting close to drawer handle
        arm_pos = states['arm_pos']
        handle_pos = states.get('drawer_handle_pos', np.zeros(3))
        
        distance = np.linalg.norm(arm_pos - handle_pos)
        if distance < 0.2:
            reward += 0.5
        if distance < 0.1:
            reward += 1.0
        
        return reward

    def _get_scene_specific_success(self, states: Dict[str, Any]) -> bool:
        """Check if drawer is successfully opened"""
        # Success if gripper is close to handle and gripper is closed
        arm_pos = states['arm_pos']
        gripper_pos = states['gripper_pos']
        handle_pos = states.get('drawer_handle_pos', np.zeros(3))
        
        distance = np.linalg.norm(arm_pos - handle_pos)
        return distance < 0.05 and gripper_pos < 0.1

    def _get_scene_specific_termination(self, states: Dict[str, Any]) -> bool:
        """Check drawer-specific termination conditions"""
        # No additional termination conditions for drawer
        return False


class CupboardEnv(AbstractMujocoEnv):
    """Environment for cupboard manipulation tasks with objects inside"""
    
    def __init__(self, show_viewer: bool = False, render_images: bool = False,
                 max_episode_steps: int = 1000, render_every_n_frames: int = 1):
        super().__init__(
            mjcf_path="env/assets/stanford_tidybot/cupboard_scene_objects_inside.xml",
            show_viewer=show_viewer,
            render_images=render_images,
            max_episode_steps=max_episode_steps,
            render_every_n_frames=render_every_n_frames
        )

    def _setup_spaces(self):
        """Set up observation and action spaces for cupboard environment"""
        # Action space: base pose (3), arm pose (3), arm quat (4), gripper (1)
        self._action_space = Box(
            low=np.array([-2.0, -2.0, -np.pi, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([2.0, 2.0, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: robot state + cube states
        obs_dict = {
            'base_pose': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'arm_quat': Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            'gripper_pos': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        }
        
        # Add cube observations
        for obj_name in ['cube1', 'cube2', 'cube3']:
            obs_dict[f'{obj_name}_pos'] = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            obs_dict[f'{obj_name}_quat'] = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self._observation_space = DictSpace(obs_dict)

    def _get_scene_specific_observation(self, states: Dict[str, Any]) -> Dict[str, Any]:
        """Get cube observations for cupboard environment"""
        observation = {}
        
        # Cube observations
        for obj_name in ['cube1', 'cube2', 'cube3']:
            if f'{obj_name}_pos' in states:
                observation[f'{obj_name}_pos'] = states[f'{obj_name}_pos'].astype(np.float32)
            else:
                observation[f'{obj_name}_pos'] = np.zeros(3, dtype=np.float32)
            
            if f'{obj_name}_quat' in states:
                observation[f'{obj_name}_quat'] = states[f'{obj_name}_quat'].astype(np.float32)
            else:
                observation[f'{obj_name}_quat'] = np.array([0., 0., 0., 1.], dtype=np.float32)
        
        return observation

    def _get_scene_specific_reward(self, states: Dict[str, Any]) -> float:
        """Calculate reward for cupboard manipulation"""
        reward = 0.0
        
        # Reward for keeping arm in reasonable workspace near cupboard
        arm_pos = states['arm_pos']
        cupboard_workspace_center = np.array([1.0, 0.0, 0.4])  # Cupboard center
        arm_distance = np.linalg.norm(arm_pos - cupboard_workspace_center)
        if arm_distance < 0.8:  # Larger workspace for cupboard
            reward += 0.1
        
        # Reward for placing objects inside cupboard
        cube1_pos = states.get('cube1_pos', np.zeros(3))
        cube2_pos = states.get('cube2_pos', np.zeros(3))
        cube3_pos = states.get('cube3_pos', np.zeros(3))
        
        # Check if cubes are inside cupboard (between x=0.8 and x=1.2, y between -0.3 and 0.3)
        cupboard_x_min, cupboard_x_max = 0.8, 1.2
        cupboard_y_min, cupboard_y_max = -0.3, 0.3
        cupboard_z_min = 0.2  # Above cupboard bottom
        
        for cube_pos in [cube1_pos, cube2_pos, cube3_pos]:
            if (cupboard_x_min < cube_pos[0] < cupboard_x_max and 
                cupboard_y_min < cube_pos[1] < cupboard_y_max and 
                cube_pos[2] > cupboard_z_min):
                reward += 1.0  # Reward for each cube inside cupboard
        
        return reward

    def _get_scene_specific_success(self, states: Dict[str, Any]) -> bool:
        """Check if objects are successfully placed inside cupboard"""
        cube1_pos = states.get('cube1_pos', np.zeros(3))
        cube2_pos = states.get('cube2_pos', np.zeros(3))
        cube3_pos = states.get('cube3_pos', np.zeros(3))
        
        # Success if at least one cube is inside cupboard
        cupboard_x_min, cupboard_x_max = 0.8, 1.2
        cupboard_y_min, cupboard_y_max = -0.3, 0.3
        cupboard_z_min = 0.2
        
        for cube_pos in [cube1_pos, cube2_pos, cube3_pos]:
            if (cupboard_x_min < cube_pos[0] < cupboard_x_max and 
                cupboard_y_min < cube_pos[1] < cupboard_y_max and 
                cube_pos[2] > cupboard_z_min):
                return True
        
        return False

    def _get_scene_specific_termination(self, states: Dict[str, Any]) -> bool:
        """Check cupboard-specific termination conditions"""
        # Terminate if robot goes too far from cupboard
        base_pose = states['base_pose']
        cupboard_center = np.array([1.0, 0.0])
        distance_from_cupboard = np.linalg.norm(base_pose[:2] - cupboard_center)
        
        if distance_from_cupboard > 2.0:  # More than 2m from cupboard
            print(f"Terminating: Robot too far from cupboard at distance {distance_from_cupboard:.3f}")
            return True
        
        # Terminate if arm goes to unsafe position
        arm_pos = states['arm_pos']
        if arm_pos[2] < -0.1:  # Below ground
            print(f"Terminating: Arm below ground at height {arm_pos[2]:.3f}")
            return True
        
        return False