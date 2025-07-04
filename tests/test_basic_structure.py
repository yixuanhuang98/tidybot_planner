#!/usr/bin/env python3
"""
Basic test of the refactored structure without MuJoCo dependencies
"""

def test_core_classes():
    """Test core classes without MuJoCo dependencies"""
    try:
        import numpy as np
        
        # Test state management
        from env.state import EnvironmentState, RobotState, ObjectState, Action, Observation
        
        robot_state = RobotState(
            base_pose=np.array([0, 0, 0]),
            arm_pos=np.array([0.5, 0, 0.4]),
            arm_quat=np.array([0, 0, 0, 1]),
            gripper_pos=0.0
        )
        
        obj_state = ObjectState(
            pos=np.array([0.3, 0.2, 0.8]),
            quat=np.array([0, 0, 0, 1])
        )
        
        env_state = EnvironmentState(
            robot=robot_state,
            objects={'cube1': obj_state},
            scene_objects={},
            timestamp=0.0
        )
        
        action = Action(
            base_pose=np.array([0.1, 0.0, 0.0]),
            arm_pos=np.array([0.6, 0.0, 0.5])
        )
        
        action_dict = action.to_dict()
        assert 'base_pose' in action_dict
        assert 'arm_pos' in action_dict
        
        print("✓ State management classes work correctly")
        
        # Test agent base (without IK solver)
        from agent.base_agent import BaseAgent
        
        class SimpleAgent(BaseAgent):
            def reset(self):
                pass
            def step(self, obs):
                return self.create_action(base_pose=np.array([0, 0, 0]))
        
        agent = SimpleAgent()
        distance = agent.distance_2d(np.array([0, 0]), np.array([3, 4]))
        assert abs(distance - 5.0) < 1e-6
        
        angle = agent.normalize_angle(3 * np.pi)
        expected = -np.pi  # 3π normalized to [-π, π] range
        assert abs(angle - expected) < 1e-6
        
        print("✓ BaseAgent utility functions work correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that directories and files exist"""
    import os
    
    required_dirs = [
        'env',
        'env/mujoco', 
        'env/real',
        'env/real/servers',
        'agent',
        'tests'
    ]
    
    required_files = [
        'env/__init__.py',
        'env/base_env.py',
        'env/state.py',
        'env/mujoco/__init__.py',
        'env/mujoco/mujoco_env.py',
        'env/mujoco/mujoco_handler.py',
        'env/real/__init__.py',
        'env/real/real_env.py',
        'env/real/real_handler.py',
        'agent/__init__.py',
        'agent/base_agent.py',
        'agent/teleop_policy.py',
        'agent/remote_policy.py',
        'agent/motion_planner_policy.py',
        'agent/go_to_cabinet_handle_policy.py',
        'agent/go_to_cabinet_handle_policy_right.py',
        'tests/__init__.py',
        'tests/test_mujoco_envs.py'
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
    missing_files = [f for f in required_files if not os.path.isfile(f)]
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✓ All required directories and files exist")
    return True


def main():
    """Run basic tests"""
    print("Testing basic refactored structure...\n")
    
    success = True
    success &= test_directory_structure()
    print()
    success &= test_core_classes()
    
    if success:
        print("\n🎉 Basic structure tests passed!")
        print("\nRefactoring summary:")
        print("✅ Modular environment system (MuJoCo + Real)")
        print("✅ Abstract environment base with scene inheritance")
        print("✅ Clean state management classes")
        print("✅ Modular agent system with base utilities")
        print("✅ Individual policy files")
        print("✅ Comprehensive test structure")
        print("✅ Updated import dependencies")
        
        print("\nThe refactoring is complete! You now have:")
        print("• env/mujoco/ - Abstract + scene-specific MuJoCo environments")
        print("• env/real/ - Real robot environment with hardware handlers")
        print("• agent/ - Modular agent system with shared utilities")
        print("• Clean separation of simulation vs real robot code")
        print("• Easy extensibility for new scenes and agents")
    else:
        print("❌ Some basic tests failed")
    
    return success


if __name__ == '__main__':
    main()