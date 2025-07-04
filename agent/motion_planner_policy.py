import math
import time
from typing import Dict, Any, Optional, List
import numpy as np

from agent.base_agent import BaseAgent


class MotionPlannerPolicy(BaseAgent):
    """Motion planner policy for pick and place tasks"""
    
    def __init__(self):
        super().__init__()
        
        # Motion planning state
        self.state = 'idle'  # States: idle, moving, manipulating, grasping
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.grasp_step = 0  # 0: position arm with open gripper, 1: close gripper
        
        # Base following parameters
        self.LOOKAHEAD_DISTANCE = 0.3  # 30 cm
        self.lookahead_position = None
        self.position_tolerance = 0.005  # 0.5 cm
        self.heading_tolerance = math.radians(2.1)  # 2.1 degrees
        
        # Object and target locations
        self.object_location = None
        self.target_location = None
        
        # Enable policy execution immediately
        self.enabled = True
        self.episode_ended = False
        
        print('Motion planner policy initialized - ready to start automatically')

    def reset(self):
        """Reset motion planning state"""
        self.state = 'idle'
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.lookahead_position = None
        self.episode_ended = False
        self.grasp_step = 0
        
        # Clean up any grasp tracking variables
        if hasattr(self, 'grasp_start_time'):
            delattr(self, 'grasp_start_time')
        if hasattr(self, 'initial_gripper_pos'):
            delattr(self, 'initial_gripper_pos')
        
        self.enabled = True
        print("Motion planner reset - starting episode automatically")

    def step(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute motion planning step"""
        if self.episode_ended or not self.enabled:
            return None
        
        return self._step(obs)

    def _step(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Main motion planning logic"""
        base_pose = obs['base_pose']
        arm_pos = obs['arm_pos'] 
        arm_quat = obs['arm_quat']
        gripper_pos = obs['gripper_pos']

        print(f"Current base pose: [{base_pose[0]:.3f}, {base_pose[1]:.3f}, {base_pose[2]:.3f}]")

        # State machine
        if self.state == 'idle':
            return self._handle_idle_state(obs, base_pose)
        elif self.state == 'moving':
            return self._handle_moving_state(obs, base_pose)
        elif self.state == 'manipulating':
            return self._handle_manipulating_state(obs, base_pose, arm_pos, arm_quat, gripper_pos)
        else:
            return self.hold_current_pose(obs)

    def _handle_idle_state(self, obs: Dict[str, Any], base_pose: np.ndarray) -> Optional[Dict[str, Any]]:
        """Handle idle state - detect objects and plan commands"""
        detected_objects = self._detect_objects_from_ground_truth(obs)
        if detected_objects:
            # Create pick command
            self.object_location = detected_objects[0]
            self.target_location = np.array([
                self.object_location[0] + 0.5,  # 50cm in X direction
                self.object_location[1],        # Same Y as object
                self.object_location[2]         # Same Z as object
            ])
            
            pick_command = {
                'primitive_name': 'pick',
                'waypoints': [base_pose[:2].tolist(), self.object_location[:2].tolist()],
                'object_3d_pos': self.object_location.copy()
            }
            
            print(f"Object detected at: {self.object_location}")
            print(f"Target placement location: {self.target_location}")
            
            # Build base command and start moving
            base_command = self._build_base_command(pick_command)
            if base_command:
                self.current_command = pick_command
                self.base_waypoints = base_command['waypoints']
                self.target_ee_pos = base_command['target_ee_pos']
                self.current_waypoint_idx = 1
                self.lookahead_position = None
                self.state = 'moving'
                print(f"Starting base movement to object")
            else:
                print("Failed to build base command")
        else:
            # No objects found, search around
            return self._search_for_objects(obs)
        
        return None

    def _handle_moving_state(self, obs: Dict[str, Any], base_pose: np.ndarray) -> Optional[Dict[str, Any]]:
        """Handle moving state - execute base movement"""
        action = self._execute_base_movement(obs)
        if action is None:  # Base movement complete
            print("Base movement complete!")
            if self.target_ee_pos is not None:
                distance_to_target = self.distance_2d(base_pose[:2], self.target_ee_pos)
                end_effector_offset = self.get_end_effector_offset(self.current_command['primitive_name'])
                diff = abs(end_effector_offset - distance_to_target)
                
                if diff < 0.002:  # 0.2 cm tolerance
                    self.state = 'manipulating'
                    print("Base reached target, starting arm manipulation")
                else:
                    print(f'Too far from target end effector position ({(100 * diff):.1f} cm)')
                    self.state = 'idle'
            else:
                self.state = 'idle'
        
        return action

    def _handle_manipulating_state(self, obs: Dict[str, Any], base_pose: np.ndarray, 
                                 arm_pos: np.ndarray, arm_quat: np.ndarray, 
                                 gripper_pos: np.ndarray) -> Optional[Dict[str, Any]]:
        """Handle manipulating state - execute arm manipulation"""
        if self.current_command['primitive_name'] == 'pick':
            return self._execute_pick_manipulation(obs, base_pose, arm_pos, gripper_pos)
        elif self.current_command['primitive_name'] == 'place':
            return self._execute_place_manipulation(obs, base_pose, arm_pos, gripper_pos)
        
        return self.hold_current_pose(obs)

    def _execute_pick_manipulation(self, obs: Dict[str, Any], base_pose: np.ndarray, 
                                 arm_pos: np.ndarray, gripper_pos: np.ndarray) -> Dict[str, Any]:
        """Execute pick manipulation sequence"""
        # Calculate object position in base frame
        if 'object_3d_pos' in self.current_command:
            object_3d_pos = self.current_command['object_3d_pos']
            object_relative_pos = self.transform_to_base_frame(object_3d_pos, base_pose)
            object_relative_pos[2] += 0.25 - 0.48  # Height adjustment
        else:
            # Fallback
            object_relative_pos = np.array([
                self.target_ee_pos[0] - base_pose[0],
                self.target_ee_pos[1] - base_pose[1], 
                -0.33
            ])
            object_relative_pos = self.transform_to_base_frame(object_relative_pos, base_pose)

        print(f"Grasp step: {self.grasp_step}")
        print(f"Object relative pos: {object_relative_pos}")

        if self.grasp_step == 0:
            # Position arm above object with open gripper
            action = self.create_action(
                base_pose=base_pose.copy(),
                arm_pos=object_relative_pos,
                arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),  # Gripper pointing down
                gripper_pos=0.0
            )
            if self.check_position_reached(arm_pos, object_relative_pos, tolerance=0.03):
                self.grasp_step = 1
                print("Arm positioned, moving to lower approach")
                
        elif self.grasp_step == 1:
            # Lower gripper closer to object
            lower_pos = object_relative_pos.copy()
            lower_pos[2] -= 0.08  # Lower by 8cm
            action = self.create_action(
                base_pose=base_pose.copy(),
                arm_pos=lower_pos,
                arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                gripper_pos=0.0
            )
            if self.check_position_reached(arm_pos, lower_pos, tolerance=0.02):
                self.grasp_step = 2
                print("Gripper lowered, closing gripper")
                
        elif self.grasp_step == 2:
            # Close gripper to grasp
            lower_pos = object_relative_pos.copy()
            lower_pos[2] -= 0.08
            action = self.create_action(
                base_pose=base_pose.copy(),
                arm_pos=lower_pos,
                arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                gripper_pos=1.0
            )
            
            # Track grasp attempt
            if not hasattr(self, 'grasp_start_time'):
                self.grasp_start_time = time.time()
                self.initial_gripper_pos = gripper_pos[0]
            
            # Check for successful grasp
            gripper_closed = gripper_pos[0] > 0.55
            gripper_progress = (gripper_pos[0] - self.initial_gripper_pos) > 0.3
            grasp_timeout = (time.time() - self.grasp_start_time) > 3.0
            
            if gripper_closed or gripper_progress or grasp_timeout:
                print("Grasp completed, moving to lift")
                self.grasp_step = 3
                delattr(self, 'grasp_start_time')
                delattr(self, 'initial_gripper_pos')
                
        elif self.grasp_step == 3:
            # Lift object
            lower_pos = object_relative_pos.copy()
            lower_pos[2] -= 0.08
            lifted_pos = lower_pos.copy()
            lifted_pos[2] += 0.28  # Lift by 28cm
            action = self.create_action(
                base_pose=base_pose.copy(),
                arm_pos=lifted_pos,
                arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                gripper_pos=1.0
            )
            if self.check_position_reached(arm_pos, lifted_pos, tolerance=0.05):
                print("Object lifted! Moving to placement location")
                # Create place command
                place_command = {
                    'primitive_name': 'place',
                    'waypoints': [base_pose[:2].tolist(), self.target_location[:2].tolist()],
                    'target_3d_pos': self.target_location.copy()
                }
                
                base_command = self._build_base_command(place_command)
                if base_command:
                    self.current_command = place_command
                    self.base_waypoints = base_command['waypoints']
                    self.target_ee_pos = base_command['target_ee_pos']
                    self.current_waypoint_idx = 1
                    self.lookahead_position = None
                    self.state = 'moving'
                    self.grasp_step = 0
                else:
                    self.episode_ended = True
                    self.state = 'idle'
        
        return action

    def _execute_place_manipulation(self, obs: Dict[str, Any], base_pose: np.ndarray,
                                  arm_pos: np.ndarray, gripper_pos: np.ndarray) -> Dict[str, Any]:
        """Execute place manipulation sequence"""
        # Calculate target position in base frame
        if 'target_3d_pos' in self.current_command:
            target_3d_pos = self.current_command['target_3d_pos']
            target_relative_pos = self.transform_to_base_frame(target_3d_pos, base_pose)
            target_relative_pos[2] += 0.10 - 0.48  # Height adjustment
        else:
            # Fallback
            target_relative_pos = np.array([
                self.target_ee_pos[0] - base_pose[0],
                self.target_ee_pos[1] - base_pose[1], 
                -0.30
            ])
            target_relative_pos = self.transform_to_base_frame(target_relative_pos, base_pose)

        print(f"Place step: {self.grasp_step}")

        if self.grasp_step == 0:
            # Position arm above placement location
            action = self.create_action(
                base_pose=base_pose.copy(),
                arm_pos=target_relative_pos,
                arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                gripper_pos=1.0
            )
            if self.check_position_reached(arm_pos, target_relative_pos, tolerance=0.05):
                self.grasp_step = 1
                print("Arm positioned above placement, opening gripper")
                
        elif self.grasp_step == 1:
            # Open gripper to place object
            action = self.create_action(
                base_pose=base_pose.copy(),
                arm_pos=target_relative_pos,
                arm_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                gripper_pos=0.0
            )
            if gripper_pos[0] < 0.2:
                print("Object placed successfully! Task complete.")
                self.episode_ended = True
                self.state = 'idle'
        
        return action

    def _execute_base_movement(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute base movement following waypoints"""
        base_pose = obs['base_pose']
        
        if self.current_waypoint_idx >= len(self.base_waypoints):
            return None  # Movement complete
        
        # Compute lookahead position
        while True:
            if self.current_waypoint_idx >= len(self.base_waypoints):
                self.lookahead_position = None
                break
                
            start = self.base_waypoints[self.current_waypoint_idx - 1]
            end = self.base_waypoints[self.current_waypoint_idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - base_pose[0], start[1] - base_pose[1])
            t2 = self.line_circle_intersection(d, f, self.LOOKAHEAD_DISTANCE)
            
            if t2 is not None:
                self.lookahead_position = [start[0] + t2 * d[0], start[1] + t2 * d[1]]
                break
            if self.current_waypoint_idx == len(self.base_waypoints) - 1:
                self.lookahead_position = None
                break
            self.current_waypoint_idx += 1
        
        # Determine target position
        if self.lookahead_position is None:
            target_position = self.base_waypoints[-1]
            position_error = self.distance_2d(base_pose[:2], target_position)
            if position_error < self.position_tolerance:
                return None  # Movement complete
        else:
            target_position = self.lookahead_position
        
        # Compute target heading
        target_heading = base_pose[2]
        if self.target_ee_pos is not None:
            dx = self.target_ee_pos[0] - base_pose[0]
            dy = self.target_ee_pos[1] - base_pose[1]
            desired_heading = math.atan2(dy, dx)
            
            frac = 1
            if self.lookahead_position is not None:
                remaining_path_length = self.LOOKAHEAD_DISTANCE
                curr_waypoint = self.lookahead_position
                for idx in range(self.current_waypoint_idx, len(self.base_waypoints)):
                    next_waypoint = self.base_waypoints[idx]
                    remaining_path_length += self.distance_2d(curr_waypoint, next_waypoint)
                    curr_waypoint = next_waypoint
                frac = math.sqrt(self.LOOKAHEAD_DISTANCE / max(remaining_path_length, self.LOOKAHEAD_DISTANCE))
            
            heading_diff = self.normalize_angle(desired_heading - base_pose[2])
            target_heading += frac * heading_diff
        
        # Create action
        return self.create_action(
            base_pose=np.array([target_position[0], target_position[1], target_heading]),
            arm_pos=obs['arm_pos'].copy(),
            arm_quat=obs['arm_quat'].copy(),
            gripper_pos=obs['gripper_pos'].copy()
        )

    def _detect_objects_from_ground_truth(self, obs: Dict[str, Any]) -> List[np.ndarray]:
        """Detect objects using ground truth and find the one with smallest x value"""
        detected_objects = []
        
        cubes = []
        for i in range(1, 4):
            cube_key = f'cube{i}_pos'
            if cube_key in obs:
                cube_pos = obs[cube_key].copy()
                cubes.append((cube_pos, i))
                print(f"Detected cube {i} at position: {cube_pos}")
        
        if cubes:
            # Sort by x position and select smallest x value
            cubes.sort(key=lambda x: x[0][0])
            target_cube_pos, target_cube_id = cubes[0]
            detected_objects.append(target_cube_pos)
            print(f"Selected cube {target_cube_id} with smallest x: {target_cube_pos[0]:.3f}")
        
        return detected_objects

    def _search_for_objects(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Search behavior when no objects detected"""
        base_pose = obs['base_pose']
        
        return self.create_action(
            base_pose=np.array([base_pose[0], base_pose[1], base_pose[2] + 0.1]),
            arm_pos=np.array([0.45, 0.0, 0.25]),
            arm_quat=np.array([0.0, 0.707, 0.0, 0.707]),
            gripper_pos=obs['gripper_pos'].copy()
        )

    def _build_base_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build base command using path planning logic"""
        assert command['primitive_name'] in {'move', 'pick', 'place', 'toss', 'shelf', 'drawer'}

        if command['primitive_name'] == 'move':
            return {'waypoints': command['waypoints'], 'target_ee_pos': None, 'position_tolerance': 0.1}

        # Modify waypoints for end effector positioning
        target_ee_pos = command['waypoints'][-1]
        end_effector_offset = self.get_end_effector_offset(command['primitive_name'])
        new_waypoint = None
        reversed_waypoints = command['waypoints'][::-1]
        
        for idx in range(1, len(reversed_waypoints)):
            start = reversed_waypoints[idx - 1]
            end = reversed_waypoints[idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - target_ee_pos[0], start[1] - target_ee_pos[1])
            t2 = self.line_circle_intersection(d, f, end_effector_offset)
            if t2 is not None:
                new_waypoint = (start[0] + t2 * d[0], start[1] + t2 * d[1])
                break
                
        if new_waypoint is not None:
            waypoints = reversed_waypoints[idx:][::-1] + [new_waypoint]
        else:
            # Base needs to back up
            print('Warning: Base needs to deviate from path')
            curr_position = command['waypoints'][0]
            signed_dist = self.distance_2d(curr_position, target_ee_pos) - end_effector_offset
            dx = target_ee_pos[0] - curr_position[0]
            dy = target_ee_pos[1] - curr_position[1]
            target_heading = self.normalize_angle(math.atan2(dy, dx))
            target_position = (curr_position[0] + signed_dist * math.cos(target_heading), 
                             curr_position[1] + signed_dist * math.sin(target_heading))
            waypoints = [curr_position, target_position]
            
        return {'waypoints': waypoints, 'target_ee_pos': target_ee_pos}