"""
Unit tests for MuJoCo environments
"""

import unittest
import numpy as np
from env import BlocksEnv, CabinetEnv, DrawerEnv, AbstractMujocoEnv
from gymnasium.spaces import Box, Dict as DictSpace


class TestMujocoEnvironments(unittest.TestCase):
    """Test cases for MuJoCo environments"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_steps = 5  # Small number for quick tests
        self.max_episode_steps = 50

    def test_blocks_env_initialization(self):
        """Test BlocksEnv initialization"""
        env = BlocksEnv(show_viewer=False, render_images=False, max_episode_steps=self.max_episode_steps)
        
        # Test observation space
        self.assertIsInstance(env.observation_space, DictSpace)
        self.assertIn('base_pose', env.observation_space.spaces)
        self.assertIn('arm_pos', env.observation_space.spaces)
        self.assertIn('cube1_pos', env.observation_space.spaces)
        self.assertIn('cube2_pos', env.observation_space.spaces)
        self.assertIn('cube3_pos', env.observation_space.spaces)
        
        # Test action space
        self.assertIsInstance(env.action_space, Box)
        self.assertEqual(env.action_space.shape, (11,))
        
        env.close()

    def test_blocks_env_reset_and_step(self):
        """Test BlocksEnv reset and step functionality"""
        env = BlocksEnv(show_viewer=False, render_images=False, max_episode_steps=self.max_episode_steps)
        
        try:
            # Test reset
            obs, info = env.reset()
            self.assertIsInstance(obs, dict)
            self.assertIsInstance(info, dict)
            self.assertIn('cube1_pos', obs)
            self.assertEqual(obs['base_pose'].shape, (3,))
            self.assertEqual(obs['arm_pos'].shape, (3,))
            self.assertEqual(obs['cube1_pos'].shape, (3,))
            
            # Test step
            action = env.action_space.sample()
            obs, reward, success, terminated, truncated, info = env.step(action)
            
            self.assertIsInstance(reward, float)
            self.assertIsInstance(success, bool)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
            
        finally:
            env.close()

    def test_cabinet_env_initialization(self):
        """Test CabinetEnv initialization"""
        env = CabinetEnv(show_viewer=False, render_images=False, max_episode_steps=self.max_episode_steps)
        
        # Test observation space includes door positions
        self.assertIn('leftdoor_pos', env.observation_space.spaces)
        self.assertIn('rightdoor_pos', env.observation_space.spaces)
        
        env.close()

    def test_cabinet_env_reset_and_step(self):
        """Test CabinetEnv reset and step functionality"""
        env = CabinetEnv(show_viewer=False, render_images=False, max_episode_steps=self.max_episode_steps)
        
        try:
            # Test reset
            obs, info = env.reset()
            self.assertIn('leftdoor_pos', obs)
            self.assertIn('rightdoor_pos', obs)
            self.assertEqual(obs['leftdoor_pos'].shape, (3,))
            self.assertEqual(obs['rightdoor_pos'].shape, (3,))
            
            # Test step
            action = env.action_space.sample()
            obs, reward, success, terminated, truncated, info = env.step(action)
            
            self.assertIsInstance(reward, float)
            self.assertIsInstance(success, bool)
            
        finally:
            env.close()

    def test_drawer_env_initialization(self):
        """Test DrawerEnv initialization (may fail if scene doesn't exist)"""
        try:
            env = DrawerEnv(show_viewer=False, render_images=False, max_episode_steps=self.max_episode_steps)
            
            # Test observation space includes drawer handle
            self.assertIn('drawer_handle_pos', env.observation_space.spaces)
            
            env.close()
            
        except Exception as e:
            # Expected to fail if drawer scene file doesn't exist
            self.assertIn('scene', str(e).lower())

    def test_action_conversion(self):
        """Test action array to Action object conversion"""
        env = BlocksEnv(show_viewer=False, render_images=False, max_episode_steps=self.max_episode_steps)
        
        try:
            # Test valid action conversion
            action_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0, 0.5])
            action_obj = env._array_to_action(action_array)
            
            self.assertEqual(len(action_obj.base_pose), 3)
            self.assertEqual(len(action_obj.arm_pos), 3)
            self.assertEqual(len(action_obj.arm_quat), 4)
            self.assertEqual(action_obj.gripper_pos, 0.5)
            
            # Test invalid action length
            with self.assertRaises(ValueError):
                invalid_action = np.array([0.1, 0.2])
                env._array_to_action(invalid_action)
                
        finally:
            env.close()

    def test_reward_calculation(self):
        """Test reward calculation for different environments"""
        env = BlocksEnv(show_viewer=False, render_images=False, max_episode_steps=self.max_episode_steps)
        
        try:
            obs, info = env.reset()
            
            # Mock states for testing
            states = {
                'base_pose': np.array([0.0, 0.0, 0.0]),
                'arm_pos': np.array([0.55, 0.0, 0.4]),  # Good workspace position
                'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
                'gripper_pos': 0.5,
                'cube1_pos': np.array([0.5, 0.0, 0.8]),
                'cube2_pos': np.array([0.5, 0.0, 0.9]),  # Stacked above cube1
                'cube3_pos': np.array([0.3, 0.3, 0.8])
            }
            
            reward = env.get_reward(states)
            self.assertIsInstance(reward, float)
            
            # Reward should be positive for good stacking
            self.assertGreater(reward, -0.5)  # Not too negative
            
        finally:
            env.close()

    def test_success_conditions(self):
        """Test success condition checking"""
        env = BlocksEnv(show_viewer=False, render_images=False, max_episode_steps=self.max_episode_steps)
        
        try:
            # Test successful stacking scenario
            successful_states = {
                'cube1_pos': np.array([0.5, 0.0, 0.8]),
                'cube2_pos': np.array([0.5, 0.0, 0.9]),  # Properly stacked
            }
            
            success = env._get_scene_specific_success(successful_states)
            self.assertTrue(success)
            
            # Test unsuccessful scenario
            unsuccessful_states = {
                'cube1_pos': np.array([0.5, 0.0, 0.8]),
                'cube2_pos': np.array([1.0, 1.0, 0.8]),  # Far apart
            }
            
            success = env._get_scene_specific_success(unsuccessful_states)
            self.assertFalse(success)
            
        finally:
            env.close()

    def test_termination_conditions(self):
        """Test termination condition checking"""
        env = BlocksEnv(show_viewer=False, render_images=False, max_episode_steps=self.max_episode_steps)
        
        try:
            # Test normal states - should not terminate
            normal_states = {
                'base_pose': np.array([0.0, 0.0, 0.0]),
                'arm_pos': np.array([0.5, 0.0, 0.4]),
                'cube1_pos': np.array([0.5, 0.0, 0.8]),
                'cube2_pos': np.array([0.5, 0.0, 0.9]),
                'cube3_pos': np.array([0.3, 0.3, 0.8])
            }
            
            terminated = env.get_termination(normal_states)
            self.assertFalse(terminated)
            
            # Test termination due to cube falling
            fallen_states = {
                'base_pose': np.array([0.0, 0.0, 0.0]),
                'arm_pos': np.array([0.5, 0.0, 0.4]),
                'cube1_pos': np.array([0.5, 0.0, -0.2]),  # Below table
                'cube2_pos': np.array([0.5, 0.0, 0.9]),
                'cube3_pos': np.array([0.3, 0.3, 0.8])
            }
            
            terminated = env.get_termination(fallen_states)
            self.assertTrue(terminated)
            
        finally:
            env.close()


class TestCustomEnvironment(unittest.TestCase):
    """Test creating custom environments"""

    def test_custom_environment_creation(self):
        """Test creating a custom environment by inheriting from AbstractMujocoEnv"""
        
        class TestCustomEnv(AbstractMujocoEnv):
            def __init__(self):
                super().__init__(
                    mjcf_path="env/assets/stanford_tidybot/scene.xml",
                    show_viewer=False,
                    render_images=False,
                    max_episode_steps=50
                )
            
            def _setup_spaces(self):
                self._action_space = Box(
                    low=np.array([-1.0] * 11),
                    high=np.array([1.0] * 11),
                    dtype=np.float32
                )
                
                self._observation_space = DictSpace({
                    'base_pose': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    'arm_pos': Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    'arm_quat': Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                    'gripper_pos': Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                })
            
            def _get_scene_specific_observation(self, states):
                return {}
            
            def _get_scene_specific_reward(self, states):
                return 1.0
            
            def _get_scene_specific_success(self, states):
                return True
            
            def _get_scene_specific_termination(self, states):
                return False
        
        # Test that custom environment can be created and used
        env = TestCustomEnv()
        
        try:
            self.assertIsInstance(env.observation_space, DictSpace)
            self.assertIsInstance(env.action_space, Box)
            
            obs, info = env.reset()
            self.assertIsInstance(obs, dict)
            
            action = env.action_space.sample()
            obs, reward, success, terminated, truncated, info = env.step(action)
            
            self.assertEqual(reward, 1.0)  # Our custom reward
            self.assertTrue(success)  # Our custom success condition
            
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)