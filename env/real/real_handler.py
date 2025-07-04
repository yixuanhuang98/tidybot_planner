import time
import numpy as np
from typing import Dict, Any, Optional

from constants import BASE_RPC_HOST, BASE_RPC_PORT, ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import BASE_CAMERA_SERIAL
from env.real.servers.arm_server import ArmManager
from env.real.servers.base_server import BaseManager
from env.real.cameras import KinovaCamera, LogitechCamera
from env.state import EnvironmentState, RobotState, ObjectState, Observation, Action


class RealHandler:
    """Real robot hardware handler"""
    
    def __init__(self, use_cameras: bool = True):
        self.use_cameras = use_cameras
        self.base = None
        self.arm = None
        self.base_camera = None
        self.wrist_camera = None
        self.current_state = EnvironmentState.create_empty()
        self.is_launched = False
        
    def launch(self):
        """Launch connection to real robot hardware"""
        if self.is_launched:
            return
            
        try:
            # Connect to base RPC server
            base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
            base_manager.connect()
            self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))
            
            # Connect to arm RPC server  
            arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
            arm_manager.connect()
            self.arm = arm_manager.Arm()
            
            # Initialize cameras if requested
            if self.use_cameras:
                self.base_camera = LogitechCamera(BASE_CAMERA_SERIAL)
                self.wrist_camera = KinovaCamera()
            
            self.is_launched = True
            print("Real robot hardware connection established")
            
        except ConnectionRefusedError as e:
            error_msg = ("Could not connect to robot RPC servers. "
                        "Make sure arm_server.py and base_server.py are running.")
            raise Exception(error_msg) from e

    def set_states(self, action: Optional[Action] = None):
        """Set states or apply action"""
        if not self.is_launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
            
        if action is None:
            # Reset robot
            self._reset_robot()
        else:
            # Execute action
            self._execute_action(action)

    def _reset_robot(self):
        """Reset the real robot to initial state"""
        print('Resetting base...')
        self.base.reset()
        
        print('Resetting arm...')
        self.arm.reset()
        
        print('Robot has been reset')
        
        # Update state after reset
        self._update_state()

    def _execute_action(self, action: Action):
        """Execute action on real robot"""
        # Convert Action object to dictionary format expected by RPC servers
        action_dict = action.to_dict()
        
        # Execute on base and arm (non-blocking)
        if not action.is_empty():
            self.base.execute_action(action_dict)
            self.arm.execute_action(action_dict)

    def step(self):
        """Step the real robot (update state)"""
        if not self.is_launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
            
        # Update current state from hardware
        self._update_state()

    def _update_state(self):
        """Update current state from real robot hardware"""
        # Get robot state from hardware
        base_state = self.base.get_state()
        arm_state = self.arm.get_state()
        
        # Create robot state
        robot_state = RobotState(
            base_pose=np.array(base_state['base_pose']),
            arm_pos=np.array(arm_state['arm_pos']),
            arm_quat=np.array(arm_state['arm_quat']),
            gripper_pos=float(arm_state['gripper_pos'])
        )
        
        # Real environment doesn't have simulated objects
        # Object states would come from perception system
        object_states = self._get_object_states_from_perception()
        
        # Scene objects (doors, drawers) would also come from perception
        scene_objects = self._get_scene_objects_from_perception()
        
        # Update environment state
        self.current_state = EnvironmentState(
            robot=robot_state,
            objects=object_states,
            scene_objects=scene_objects,
            timestamp=time.time()
        )

    def _get_object_states_from_perception(self) -> Dict[str, ObjectState]:
        """Get object states from perception system (placeholder)"""
        # TODO: Implement real perception system
        # For now, return empty dict - would be filled by vision system
        return {}

    def _get_scene_objects_from_perception(self) -> Dict[str, Any]:
        """Get scene object states from perception (placeholder)"""
        # TODO: Implement real perception for doors, drawers, etc.
        # For now, return empty dict - would be filled by vision system
        return {}

    def get_states(self) -> Dict[str, Any]:
        """Get current states as dictionary"""
        return {
            'base_pose': self.current_state.robot.base_pose,
            'arm_pos': self.current_state.robot.arm_pos,
            'arm_quat': self.current_state.robot.arm_quat,
            'gripper_pos': self.current_state.robot.gripper_pos,
            **{f'{obj_name}_pos': obj_state.pos for obj_name, obj_state in self.current_state.objects.items()},
            **{f'{obj_name}_quat': obj_state.quat for obj_name, obj_state in self.current_state.objects.items()},
            **self.current_state.scene_objects
        }

    def get_extra(self) -> Dict[str, Any]:
        """Get extra information including camera images"""
        extra = {'timestamp': self.current_state.timestamp}
        
        # Add camera images if available
        if self.use_cameras and self.base_camera and self.wrist_camera:
            try:
                extra['base_image'] = self.base_camera.get_image()
                extra['wrist_image'] = self.wrist_camera.get_image()
            except Exception as e:
                print(f"Warning: Failed to get camera images: {e}")
        
        return extra

    def render(self):
        """Render the real environment (return camera images)"""
        if self.use_cameras and self.base_camera and self.wrist_camera:
            try:
                return {
                    'base_image': self.base_camera.get_image(),
                    'wrist_image': self.wrist_camera.get_image()
                }
            except Exception as e:
                print(f"Warning: Failed to render camera images: {e}")
                return {}
        return {}

    def close(self):
        """Close connections to real robot hardware"""
        if self.base:
            try:
                self.base.close()
            except:
                pass
            
        if self.arm:
            try:
                self.arm.close()
            except:
                pass
            
        if self.base_camera:
            try:
                self.base_camera.close()
            except:
                pass
            
        if self.wrist_camera:
            try:
                self.wrist_camera.close()
            except:
                pass
        
        self.is_launched = False
        print("Real robot hardware connections closed")