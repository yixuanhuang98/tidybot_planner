from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np


@dataclass
class RobotState:
    """Robot state containing base, arm, and gripper information"""
    base_pose: np.ndarray  # [x, y, theta] - base position and orientation
    arm_pos: np.ndarray    # [x, y, z] - end effector position in base frame
    arm_quat: np.ndarray   # [x, y, z, w] - end effector orientation quaternion
    gripper_pos: float     # gripper opening (0.0 = closed, 1.0 = open)
    
    def __post_init__(self):
        self.base_pose = np.array(self.base_pose)
        self.arm_pos = np.array(self.arm_pos)
        self.arm_quat = np.array(self.arm_quat)
        self.gripper_pos = float(self.gripper_pos)


@dataclass
class ObjectState:
    """Object state with position and orientation"""
    pos: np.ndarray   # [x, y, z] - object position
    quat: np.ndarray  # [x, y, z, w] - object orientation quaternion
    
    def __post_init__(self):
        self.pos = np.array(self.pos)
        self.quat = np.array(self.quat)


@dataclass
class EnvironmentState:
    """Complete environment state including robot and objects"""
    robot: RobotState
    objects: Dict[str, ObjectState]  # object_name -> ObjectState
    scene_objects: Dict[str, Any]    # scene elements like door positions
    timestamp: float
    
    def __post_init__(self):
        self.timestamp = float(self.timestamp)
    
    @classmethod
    def create_empty(cls):
        """Create an empty environment state"""
        return cls(
            robot=RobotState(
                base_pose=np.zeros(3),
                arm_pos=np.zeros(3),
                arm_quat=np.array([0., 0., 0., 1.]),
                gripper_pos=0.0
            ),
            objects={},
            scene_objects={},
            timestamp=0.0
        )


class Observation:
    """Environment observation containing all sensor data"""
    
    def __init__(self, state: EnvironmentState, images: Optional[Dict[str, np.ndarray]] = None):
        self.state = state
        self.images = images or {}
        
    def get_robot_obs(self) -> Dict[str, Any]:
        """Get robot-specific observations"""
        return {
            'base_pose': self.state.robot.base_pose.copy(),
            'arm_pos': self.state.robot.arm_pos.copy(),
            'arm_quat': self.state.robot.arm_quat.copy(),
            'gripper_pos': self.state.robot.gripper_pos,
        }
    
    def get_object_obs(self) -> Dict[str, Any]:
        """Get object observations"""
        obs = {}
        for obj_name, obj_state in self.state.objects.items():
            obs[f'{obj_name}_pos'] = obj_state.pos.copy()
            obs[f'{obj_name}_quat'] = obj_state.quat.copy()
        return obs
    
    def get_scene_obs(self) -> Dict[str, Any]:
        """Get scene observations"""
        return {k: v.copy() if isinstance(v, np.ndarray) else v 
                for k, v in self.state.scene_objects.items()}
    
    def get_image_obs(self) -> Dict[str, np.ndarray]:
        """Get image observations"""
        return {k: v.copy() for k, v in self.images.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary format"""
        obs = {}
        obs.update(self.get_robot_obs())
        obs.update(self.get_object_obs())
        obs.update(self.get_scene_obs())
        obs.update(self.get_image_obs())
        return obs


class Action:
    """Environment action containing robot commands"""
    
    def __init__(self, 
                 base_pose: Optional[np.ndarray] = None,
                 arm_pos: Optional[np.ndarray] = None,
                 arm_quat: Optional[np.ndarray] = None,
                 gripper_pos: Optional[float] = None):
        self.base_pose = base_pose
        self.arm_pos = arm_pos
        self.arm_quat = arm_quat
        self.gripper_pos = gripper_pos
    
    @classmethod
    def from_dict(cls, action_dict: Dict[str, Any]) -> 'Action':
        """Create action from dictionary"""
        return cls(
            base_pose=action_dict.get('base_pose'),
            arm_pos=action_dict.get('arm_pos'),
            arm_quat=action_dict.get('arm_quat'),
            gripper_pos=action_dict.get('gripper_pos')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary format"""
        action_dict = {}
        if self.base_pose is not None:
            action_dict['base_pose'] = self.base_pose
        if self.arm_pos is not None:
            action_dict['arm_pos'] = self.arm_pos
        if self.arm_quat is not None:
            action_dict['arm_quat'] = self.arm_quat
        if self.gripper_pos is not None:
            action_dict['gripper_pos'] = self.gripper_pos
        return action_dict
    
    def is_empty(self) -> bool:
        """Check if action contains any commands"""
        return all(v is None for v in [self.base_pose, self.arm_pos, self.arm_quat, self.gripper_pos])