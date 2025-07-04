#!/usr/bin/env python3
"""
Quick test to verify the refactored structure works
"""

def test_imports():
    """Test that all the new modules can be imported"""
    try:
        # Test environment imports
        from env.state import EnvironmentState, RobotState, ObjectState, Action, Observation
        print("✓ State classes imported successfully")
        
        from env.base_env import BaseEnv
        print("✓ BaseEnv imported successfully")
        
        from env.mujoco.mujoco_env import AbstractMujocoEnv, BlocksEnv, CabinetEnv, DrawerEnv
        print("✓ MuJoCo environments imported successfully")
        
        # Test agent imports
        from agent.base_agent import BaseAgent
        print("✓ BaseAgent imported successfully")
        
        from agent.teleop_policy import TeleopPolicy
        print("✓ TeleopPolicy imported successfully")
        
        from agent.remote_policy import RemotePolicy
        print("✓ RemotePolicy imported successfully")
        
        from agent.motion_planner_policy import MotionPlannerPolicy
        print("✓ MotionPlannerPolicy imported successfully")
        
        from agent.go_to_cabinet_handle_policy import GoToCabinetHandlePolicy
        print("✓ GoToCabinetHandlePolicy imported successfully")
        
        from agent.go_to_cabinet_handle_policy_right import GoToCabinetHandlePolicyRight
        print("✓ GoToCabinetHandlePolicyRight imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_state_classes():
    """Test the state management classes"""
    try:
        import numpy as np
        from env.state import EnvironmentState, RobotState, ObjectState, Action
        
        # Test RobotState
        robot_state = RobotState(
            base_pose=np.array([0, 0, 0]),
            arm_pos=np.array([0.5, 0, 0.4]),
            arm_quat=np.array([0, 0, 0, 1]),
            gripper_pos=0.0
        )
        print("✓ RobotState created successfully")
        
        # Test ObjectState
        obj_state = ObjectState(
            pos=np.array([0.3, 0.2, 0.8]),
            quat=np.array([0, 0, 0, 1])
        )
        print("✓ ObjectState created successfully")
        
        # Test EnvironmentState
        env_state = EnvironmentState(
            robot=robot_state,
            objects={'cube1': obj_state},
            scene_objects={},
            timestamp=0.0
        )
        print("✓ EnvironmentState created successfully")
        
        # Test Action
        action = Action(
            base_pose=np.array([0.1, 0.0, 0.0]),
            arm_pos=np.array([0.6, 0.0, 0.5]),
            arm_quat=np.array([0, 0, 0, 1]),
            gripper_pos=0.5
        )
        print("✓ Action created successfully")
        
        print("✓ All state classes work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ State class error: {e}")
        return False


def test_base_agent():
    """Test the base agent functionality"""
    try:
        import numpy as np
        from agent.base_agent import BaseAgent
        
        # Create a dummy agent class
        class TestAgent(BaseAgent):
            def reset(self):
                pass
            
            def step(self, obs):
                return self.hold_current_pose(obs)
        
        agent = TestAgent()
        
        # Test utility functions
        distance = agent.distance_2d(np.array([0, 0]), np.array([3, 4]))
        assert abs(distance - 5.0) < 1e-6, "Distance calculation failed"
        print("✓ Distance calculation works")
        
        # Test angle normalization
        angle = agent.normalize_angle(3 * np.pi)
        assert abs(angle - (-np.pi)) < 1e-6, f"Angle normalization failed: {angle} vs {-np.pi}"
        print("✓ Angle normalization works")
        
        # Test action creation
        action = agent.create_action(
            base_pose=np.array([0, 0, 0]),
            arm_pos=np.array([0.5, 0, 0.4]),
            gripper_pos=0.5
        )
        assert 'base_pose' in action
        assert 'arm_pos' in action
        assert 'gripper_pos' in action
        print("✓ Action creation works")
        
        print("✓ BaseAgent functionality verified!")
        return True
        
    except Exception as e:
        print(f"❌ BaseAgent error: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing refactored codebase structure...\n")
    
    success = True
    success &= test_imports()
    print()
    success &= test_state_classes()
    print()
    success &= test_base_agent()
    
    print(f"\n{'🎉 All tests passed!' if success else '❌ Some tests failed'}")
    return success


if __name__ == '__main__':
    main()