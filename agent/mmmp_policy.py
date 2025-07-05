import math
import time
import numpy as np
from enum import Enum

# Import BaseAgent base class
from .base_agent import BaseAgent
from .planning_utils import dot, intersect, distance, restrict_heading_range


class PickState(Enum):
    APPROACH = "approach"
    LOWER = "lower"
    GRASP = "grasp"
    LIFT = "lift"


class PlaceState(Enum):
    APPROACH = "approach"
    RELEASE = "release"


class ManipulatePrimitive:
    """Class to handle individual manipulation phases"""
    
    def __init__(self, policy_instance):
        self.policy = policy_instance
        
    def _extract_observations(self, obs):
        """Extract common observation values"""
        return (
            obs['base_pose'],
            obs['arm_pos'], 
            obs['arm_quat'],
            obs['gripper_pos']
        )
    
    def _calculate_target_relative_pos(self, obs, target_3d_pos, height_offset):
        """Calculate target position in base's local coordinate frame"""
        base_pose = obs['base_pose']
        
        # Calculate global position difference
        global_diff = np.array([
            target_3d_pos[0] - base_pose[0],
            target_3d_pos[1] - base_pose[1], 
            target_3d_pos[2] + height_offset - self.policy.ROBOT_BASE_HEIGHT
        ])
        
        # Transform to base's local coordinate frame
        base_angle = base_pose[2]
        cos_angle = math.cos(-base_angle)
        sin_angle = math.sin(-base_angle)
        
        target_relative_pos = np.array([
            cos_angle * global_diff[0] - sin_angle * global_diff[1],
            sin_angle * global_diff[0] + cos_angle * global_diff[1],
            global_diff[2]
        ])
        
        return target_relative_pos
        
    def _get_gripper_down_quat(self):
        """Get standard gripper down quaternion"""
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def approaching(self, obs, target_3d_pos):
        """Phase 1: Moving end effector to target pose (approach)"""
        base_pose, arm_pos, arm_quat, gripper_pos = self._extract_observations(obs)
        
        # Calculate target position with approach height offset
        target_relative_pos = self._calculate_target_relative_pos(
            obs, target_3d_pos, self.policy.PICK_APPROACH_HEIGHT_OFFSET
        )
        
        # Position arm above object with open gripper (safe approach)
        target_arm_pos = target_relative_pos
        target_arm_quat = self._get_gripper_down_quat()
        target_gripper_pos = np.array([0.0])  # Gripper open

        print(f"Approaching: Positioning arm above object with open gripper")
        
        # Check if positioned correctly
        phase_complete = np.allclose(arm_pos, target_arm_pos, atol=0.03)  # 3cm tolerance
        if phase_complete:
            print("Arm positioned above object, moving to lower approach")
            
        return target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete
    
    def topgrasp(self, obs, target_3d_pos):
        """Phase 2: Lowering gripper closer to object for precise grasping"""
        base_pose, arm_pos, arm_quat, gripper_pos = self._extract_observations(obs)
        
        # Calculate target position with approach height offset, then lower it
        target_relative_pos = self._calculate_target_relative_pos(
            obs, target_3d_pos, self.policy.PICK_APPROACH_HEIGHT_OFFSET
        )
        
        # Lower gripper closer to object
        target_arm_pos = target_relative_pos.copy()
        target_arm_pos[2] -= self.policy.PICK_LOWER_DIST  # Lower by 8cm
        target_arm_quat = self._get_gripper_down_quat()
        target_gripper_pos = np.array([0.0])  # Gripper open

        print(f"TopGrasp: Lowering gripper for precise approach... target: {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}")
        
        # Check if positioned correctly
        phase_complete = np.allclose(arm_pos, target_arm_pos, atol=0.02)  # 2cm tolerance
        if phase_complete:
            print("Gripper lowered to grasping position, closing gripper")
            
        return target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete
    
    def immobilizing(self, obs, target_3d_pos):
        """Phase 3: Close gripper to grasp object"""
        base_pose, arm_pos, arm_quat, gripper_pos = self._extract_observations(obs)
        
        # Calculate target position with approach height offset, then lower it
        target_relative_pos = self._calculate_target_relative_pos(
            obs, target_3d_pos, self.policy.PICK_APPROACH_HEIGHT_OFFSET
        )
        
        # Maintain lowered position and close gripper
        target_arm_pos = target_relative_pos.copy()
        target_arm_pos[2] -= self.policy.PICK_LOWER_DIST
        target_arm_quat = self._get_gripper_down_quat()
        target_gripper_pos = np.array([1.0])  # Close gripper

        print(f"Immobilizing: Closing gripper... current position: {gripper_pos[0]:.3f}")
        
        # Initialize grasp attempt tracking
        if not hasattr(self.policy, 'grasp_start_time'):
            self.policy.grasp_start_time = time.time()
            self.policy.initial_gripper_pos = gripper_pos[0]
            print(f"Started grasp attempt, initial gripper pos: {self.policy.initial_gripper_pos:.3f}")
        
        # Check for successful grasp (multiple criteria)
        gripper_closed_enough = gripper_pos[0] > self.policy.GRASP_SUCCESS_THRESHOLD
        gripper_progress = (gripper_pos[0] - self.policy.initial_gripper_pos) > self.policy.GRASP_PROGRESS_THRESHOLD
        grasp_timeout = (time.time() - self.policy.grasp_start_time) > self.policy.GRASP_TIMEOUT_S
        
        phase_complete = gripper_closed_enough or gripper_progress or grasp_timeout
        if phase_complete:
            if gripper_closed_enough or gripper_progress:
                print(f"Grasp successful! Gripper pos: {gripper_pos[0]:.3f}, progress: {gripper_pos[0] - self.policy.initial_gripper_pos:.3f}")
            else:
                print(f"Grasp timeout reached, proceeding with current grip: {gripper_pos[0]:.3f}")
            
            # Clean up tracking variables
            if hasattr(self.policy, 'grasp_start_time'):
                delattr(self.policy, 'grasp_start_time')
            if hasattr(self.policy, 'initial_gripper_pos'):
                delattr(self.policy, 'initial_gripper_pos')
            print("Moving to lift phase!")
            
        return target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete
    
    def transporting(self, obs, target_3d_pos):
        """Phase 4: Moving end effector with closed gripper to target pose (lift)"""
        base_pose, arm_pos, arm_quat, gripper_pos = self._extract_observations(obs)
        
        # Calculate target position with approach height offset
        target_relative_pos = self._calculate_target_relative_pos(
            obs, target_3d_pos, self.policy.PICK_APPROACH_HEIGHT_OFFSET
        )
        
        # Lift object from the grasping position
        lifted_pos = target_relative_pos.copy()
        lifted_pos[2] += (self.policy.PICK_LIFT_DIST - self.policy.PICK_LOWER_DIST)  # Net lift
        target_arm_pos = lifted_pos
        target_arm_quat = self._get_gripper_down_quat()
        target_gripper_pos = np.array([1.0])  # Gripper closed

        print(f"Transporting: Lifting object... target height: {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}")
        
        # Check if lifted correctly
        phase_complete = np.allclose(arm_pos, target_arm_pos, atol=0.05)  # 5cm tolerance
        if phase_complete:
            print("Object lifted successfully!")
            
        return target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete
    
    def releasing(self, obs, target_3d_pos):
        """Phase 5: Open gripper to release object"""
        base_pose, arm_pos, arm_quat, gripper_pos = self._extract_observations(obs)
        
        # Calculate target position with placement height offset
        target_relative_pos = self._calculate_target_relative_pos(
            obs, target_3d_pos, self.policy.PLACE_APPROACH_HEIGHT_OFFSET
        )
        
        # Open gripper to place object
        target_arm_pos = target_relative_pos  # Maintain arm position
        target_arm_quat = self._get_gripper_down_quat()
        target_gripper_pos = np.array([0.0])  # Gripper open

        print(f"Releasing: Opening gripper... current position: {gripper_pos[0]:.3f}")
        
        # Check if gripper is open
        phase_complete = gripper_pos[0] < self.policy.PLACE_SUCCESS_THRESHOLD
        if phase_complete:
            print("Object placed successfully! Task complete.")
            
        return target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete


# Modified Motion Planner Policy with strict 0.55m positioning and sequential base-arm execution
class MMMPPolicy(BaseAgent):
    # Base following parameters (from BaseController)
    LOOKAHEAD_DISTANCE = 0.3  # 30 cm
    POSITION_TOLERANCE = 0.005  # 0.5 cm (reduced from 1.5 cm)
    HEADING_TOLERANCE = math.radians(2.1)  # 2.1 degrees
    GRASP_BASE_TOLERANCE = 0.002  # 0.2 cm for grasp
    PLACE_BASE_TOLERANCE = 0.02   # 1.0 cm for placement
    
    # Object and target locations
    PLACEMENT_X_OFFSET = 0.5  # 50cm in X direction

    # Manipulation parameters
    ROBOT_BASE_HEIGHT = 0.48
    PICK_APPROACH_HEIGHT_OFFSET = 0.25
    PICK_LOWER_DIST = 0.08
    PICK_LIFT_DIST = 0.28  # Net lift is (PICK_LIFT_DIST - PICK_LOWER_DIST)
    PLACE_APPROACH_HEIGHT_OFFSET = 0.10

    # Fixed end effector offset - always 0.55m for consistent positioning
    FIXED_END_EFFECTOR_OFFSET = 0.55

    # Grasping parameters
    GRASP_SUCCESS_THRESHOLD = 0.7
    GRASP_PROGRESS_THRESHOLD = 0.3
    GRASP_TIMEOUT_S = 3.0
    PLACE_SUCCESS_THRESHOLD = 0.2

    def __init__(self, socketio=None):
        # Motion planning state - following controller.py pattern
        self.state = 'idle'  # States: idle, moving, base_stopping, manipulating
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.grasp_state = None
        
        # Base following parameters
        self.lookahead_position = None
        
        # Object and target locations (using ground truth from MuJoCo)
        self.object_location = None
        self.target_location = None
        
        # Enable policy execution immediately (no web interface required)
        self.enabled = True
        self.episode_ended = False
        
        # Initialize manipulation primitive handler
        self.manipulate_primitive = ManipulatePrimitive(self)
        
        # Socket.IO for web interface updates
        self.socketio = socketio
        
        print(f'MMMP policy initialized - base will position at exactly {self.FIXED_END_EFFECTOR_OFFSET}m from target')

    def emit_robot_status(self, obs, current_instruction=""):
        """Emit robot status to web interface"""
        if self.socketio is None:
            return
            
        try:
            base_pose = obs['base_pose']
            
            # Safely convert to list if it's a numpy array
            def safe_to_list(data):
                if hasattr(data, 'tolist'):
                    return data.tolist()
                elif isinstance(data, (list, tuple)):
                    return list(data)
                else:
                    return data
            
            # Safely get grasp state string
            grasp_state_str = 'IDLE'
            if self.grasp_state is not None:
                if hasattr(self.grasp_state, 'value'):
                    grasp_state_str = str(self.grasp_state.value)
                else:
                    grasp_state_str = str(self.grasp_state)
            
            # Prepare status data
            status_data = {
                'target_ee_pos': safe_to_list(self.target_ee_pos) if self.target_ee_pos is not None else [0.0, 0.0],
                'base_pose': safe_to_list(base_pose),
                'grasp_state': grasp_state_str,
                'robot_state': self.state,
                'instruction': current_instruction,
                'task_complete': self.episode_ended,
                'completion_message': 'Object placed successfully! Task complete.' if self.episode_ended else None
            }
            
            # Emit to all connected clients
            self.socketio.emit('robot_status', status_data)
            
        except Exception as e:
            print(f"Error emitting robot status: {e}")

    def reset(self):
        # Reset motion planning state
        self.state = 'idle'
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.lookahead_position = None
        self.episode_ended = False
        self.grasp_state = None
        
        # Clean up any grasp tracking variables
        if hasattr(self, 'grasp_start_time'):
            delattr(self, 'grasp_start_time')
        if hasattr(self, 'initial_gripper_pos'):
            delattr(self, 'initial_gripper_pos')
        
        # Enable policy execution immediately
        self.enabled = True
        
        print("MMMP policy reset - starting episode automatically")

    def step(self, obs):
        # Return no action if episode has ended
        if self.episode_ended:
            return None

        # Return no action if robot is not enabled
        if not self.enabled:
            return None

        return self._step(obs)

    def _step(self, obs):
        # Extract current state
        base_pose = obs['base_pose']
        arm_pos = obs['arm_pos'] 
        arm_quat = obs['arm_quat']
        gripper_pos = obs['gripper_pos']

        # Debug: Print current base pose
        print(f"Current base pose: [{base_pose[0]:.3f}, {base_pose[1]:.3f}, {base_pose[2]:.3f}]")

        # Emit current robot status
        current_instruction = "Waiting for robot to start..."

        # State machine with strict base-arm sequencing
        if self.state == 'idle':
            current_instruction = "Detecting objects and planning task..."
            # Detect objects and plan new command
            detected_objects = self.detect_objects_from_ground_truth(obs)
            if not detected_objects:
                self.emit_robot_status(obs, current_instruction)
                return None
            
            self.object_location = detected_objects[0]
            # Set placement location relative to detected object (e.g., 50cm away)
            self.target_location = np.array([
                self.object_location[0] + self.PLACEMENT_X_OFFSET,  # 50cm in X direction
                self.object_location[1],        # Same Y as object
                self.object_location[2]         # Same Z as object (table height)
            ])
            
            pick_command = {
                'primitive_name': 'pick',
                'waypoints': [base_pose[:2].tolist(), self.object_location[:2].tolist()],
                'object_3d_pos': self.object_location.copy()  # Store full 3D position
            }
            
            print(f"Object detected at: {self.object_location}")
            print(f"Target placement location: {self.target_location}")
            print(f"Creating pick command with waypoints: {pick_command['waypoints']}")
            
            # Build base command and start moving
            base_command = self.build_base_command(pick_command)

            print(f"Base command: {base_command}")
            
            if base_command:
                self.current_command = pick_command
                self.base_waypoints = base_command['waypoints']
                self.target_ee_pos = base_command['target_ee_pos']
                self.current_waypoint_idx = 1
                self.lookahead_position = None
                self.state = 'moving'
                current_instruction = f"Moving base to object at {self.object_location[:2]}"
                print(f"Starting base movement to object at {self.object_location}")
                print(f"Base waypoints: {self.base_waypoints}")
                print(f"Target EE position: {self.target_ee_pos}")
                print(f"Will position base at exactly {self.FIXED_END_EFFECTOR_OFFSET}m from target")
            else:
                current_instruction = "Failed to build base command"
                print("Failed to build base command")

                
        elif self.state == 'moving':
            current_instruction = f"Moving base to target position..."
            # Execute base movement following waypoints
            action = self.execute_base_movement(obs)
            if action is None:  # Base movement complete
                print("Base movement complete! Starting arm manipulation...")
                # Transition directly to manipulating state
                self.state = 'manipulating'
                current_instruction = "Base movement complete, starting arm manipulation..."
            else:
                # Print target base pose from action
                if action and 'base_pose' in action:
                    target_pose = action['base_pose']
                    print(f"Target base pose: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
            
            self.emit_robot_status(obs, current_instruction)
            return action
            
        elif self.state == 'manipulating':
            # Execute arm manipulation using ManipulatePrimitive
            action = self.execute_manipulation(obs)
            # Status will be emitted within execute_manipulation method
            return action

        # Emit status for idle and other states
        self.emit_robot_status(obs, current_instruction)

        # Default: hold current pose
        action = {
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos.copy(),
            'arm_quat': arm_quat.copy(),
            'gripper_pos': gripper_pos.copy(),
        }
        print(f"Default action - holding current pose")
        return action

    def execute_manipulation(self, obs):
        """Execute arm manipulation using ManipulatePrimitive class"""
        base_pose = obs['base_pose']
        arm_pos = obs['arm_pos']
        arm_quat = obs['arm_quat']
        gripper_pos = obs['gripper_pos']
        
        current_instruction = "Unknown manipulation state"
        
        # Initialize target variables to current state as defaults
        target_arm_pos = arm_pos.copy()
        target_arm_quat = arm_quat.copy()
        target_gripper_pos = gripper_pos.copy()
        
        if self.current_command['primitive_name'] == 'pick':
            if self.grasp_state is None:
                self.grasp_state = PickState.APPROACH

            object_3d_pos = self.current_command['object_3d_pos']
            
            print(f"Arm manipulation: Base angle: {base_pose[2]:.3f} rad ({math.degrees(base_pose[2]):.1f} deg)")
            print(f"Object global pos: {object_3d_pos}")
            print(f"Current arm pos: {arm_pos}")
            print(f"Grasp state: {self.grasp_state}")
            
            if self.grasp_state == PickState.APPROACH:
                current_instruction = "Approaching: Positioning arm above object with open gripper"
                target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete = \
                    self.manipulate_primitive.approaching(obs, object_3d_pos)
                if phase_complete:
                    self.grasp_state = PickState.LOWER
                    
            elif self.grasp_state == PickState.LOWER:
                current_instruction = "TopGrasp: Lowering gripper for precise approach"
                target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete = \
                    self.manipulate_primitive.topgrasp(obs, object_3d_pos)
                if phase_complete:
                    self.grasp_state = PickState.GRASP
                    
            elif self.grasp_state == PickState.GRASP:
                current_instruction = f"Immobilizing: Closing gripper... current position: {gripper_pos[0]:.3f}"
                target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete = \
                    self.manipulate_primitive.immobilizing(obs, object_3d_pos)
                if phase_complete:
                    self.grasp_state = PickState.LIFT
                    
            elif self.grasp_state == PickState.LIFT:
                target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete = \
                    self.manipulate_primitive.transporting(obs, object_3d_pos)
                current_instruction = f"Transporting: Lifting object... target height: {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}"
                if phase_complete:
                    print("Object lifted successfully! Now moving to placement location.")
                    current_instruction = "Object lifted successfully! Moving to placement location."
                    # Create place command
                    place_command = {
                        'primitive_name': 'place',
                        'waypoints': [base_pose[:2].tolist(), self.target_location[:2].tolist()],
                        'target_3d_pos': self.target_location.copy()
                    }
                    
                    base_command = self.build_base_command(place_command)
                    if base_command:
                        self.current_command = place_command
                        self.base_waypoints = base_command['waypoints']
                        self.target_ee_pos = base_command['target_ee_pos']
                        self.current_waypoint_idx = 1
                        self.lookahead_position = None
                        self.state = 'moving'
                        self.grasp_state = None  # Reset for next manipulation
                        print(f"Starting base movement to placement location at {self.target_location}")
                    else:
                        print("Failed to build place command")
                        self.episode_ended = True
                        self.state = 'idle'
                        current_instruction = "Failed to build place command - task ended"

        elif self.current_command['primitive_name'] == 'place':
            if self.grasp_state is None:
                self.grasp_state = PlaceState.APPROACH

            target_3d_pos = self.current_command['target_3d_pos']
            
            print(f"Placing: Target EE pos: {self.target_ee_pos}, Base pose: {base_pose}")
            print(f"Grasp state: {self.grasp_state}")
            
            if self.grasp_state == PlaceState.APPROACH:
                current_instruction = "Approaching placement location with closed gripper"
                target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete = \
                    self.manipulate_primitive.approaching(obs, target_3d_pos)
                if phase_complete:
                    self.grasp_state = PlaceState.RELEASE
                    print("Arm positioned above placement location, opening gripper")
                    
            elif self.grasp_state == PlaceState.RELEASE:
                current_instruction = f"Releasing: Opening gripper... current position: {gripper_pos[0]:.3f}"
                target_arm_pos, target_arm_quat, target_gripper_pos, phase_complete = \
                    self.manipulate_primitive.releasing(obs, target_3d_pos)
                if phase_complete:
                    self.episode_ended = True  # End the episode
                    self.state = 'idle'
                    current_instruction = "Object placed successfully! Task complete."
        
        # Emit robot status with current instruction
        self.emit_robot_status(obs, current_instruction)
        
        # Create action from targets (hold base pose steady during arm manipulation)
        action = {
            'base_pose': base_pose.copy(),  # Keep base stationary
            'arm_pos': target_arm_pos,
            'arm_quat': target_arm_quat,
            'gripper_pos': target_gripper_pos,
        }
        return action

    def execute_base_movement(self, obs):
        """Execute base movement following waypoints like BaseController"""
        base_pose = obs['base_pose']
        
        # Check if we've reached the final waypoint
        if self.current_waypoint_idx >= len(self.base_waypoints):
            print("All waypoints completed")
            return None  # Movement complete
        
        print(f"Current waypoint index: {self.current_waypoint_idx}/{len(self.base_waypoints)}")
        print(f"Base waypoints: {self.base_waypoints}")
        
        # Compute lookahead position (simplified version of BaseController logic)
        while True:
            if self.current_waypoint_idx >= len(self.base_waypoints):
                self.lookahead_position = None
                print("Reached end of waypoints, no lookahead")
                break
                
            start = self.base_waypoints[self.current_waypoint_idx - 1]
            end = self.base_waypoints[self.current_waypoint_idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - base_pose[0], start[1] - base_pose[1])
            t2 = intersect(d, f, self.LOOKAHEAD_DISTANCE)
            
            print(f"Waypoint {self.current_waypoint_idx}: start={start}, end={end}")
            print(f"d={d}, f={f}, t2={t2}")
            
            if t2 is not None:
                self.lookahead_position = [start[0] + t2 * d[0], start[1] + t2 * d[1]]
                print(f"Lookahead position: {self.lookahead_position}")
                break
            if self.current_waypoint_idx == len(self.base_waypoints) - 1:
                self.lookahead_position = None
                print("Last waypoint, no lookahead")
                break
            print(f"Moving to next waypoint: {self.current_waypoint_idx + 1}")
            self.current_waypoint_idx += 1
        
        # Determine target position
        if self.lookahead_position is None:
            target_position = self.base_waypoints[-1]
            print(f"Using final waypoint as target: {target_position}")
            # Check if we've reached the final position
            position_error = distance(base_pose[:2], target_position)
            print(f"Position error to final target: {position_error:.3f} (tolerance: {self.POSITION_TOLERANCE})")
            if position_error < self.POSITION_TOLERANCE:
                print("Reached final position within tolerance")
                return None  # Movement complete
        else:
            target_position = self.lookahead_position
            print(f"Using lookahead as target: {target_position}")
        
        # Compute target heading
        target_heading = base_pose[2]
        if self.target_ee_pos is not None:
            # Turn to face target end effector position
            dx = self.target_ee_pos[0] - base_pose[0]
            dy = self.target_ee_pos[1] - base_pose[1]
            desired_heading = math.atan2(dy, dx)
            
            print(f"Target EE: {self.target_ee_pos}, dx={dx:.3f}, dy={dy:.3f}, desired_heading={desired_heading:.3f}")
            
            frac = 1
            if self.lookahead_position is not None:
                # Turn slowly at first, more quickly as we approach
                remaining_path_length = self.LOOKAHEAD_DISTANCE
                curr_waypoint = self.lookahead_position
                for idx in range(self.current_waypoint_idx, len(self.base_waypoints)):
                    next_waypoint = self.base_waypoints[idx]
                    remaining_path_length += distance(curr_waypoint, next_waypoint)
                    curr_waypoint = next_waypoint
                frac = math.sqrt(self.LOOKAHEAD_DISTANCE / max(remaining_path_length, self.LOOKAHEAD_DISTANCE))
            
            heading_diff = restrict_heading_range(desired_heading - base_pose[2])
            target_heading += frac * heading_diff
            print(f"Heading: current={base_pose[2]:.3f}, desired={desired_heading:.3f}, diff={heading_diff:.3f}, frac={frac:.3f}, target={target_heading:.3f}")
        
        # Create action to move towards target
        action = {
            'base_pose': np.array([target_position[0], target_position[1], target_heading]),
            'arm_pos': obs['arm_pos'].copy(),
            'arm_quat': obs['arm_quat'].copy(),
            'gripper_pos': obs['gripper_pos'].copy(),
        }
        
        return action

    def find_base_heading(self, base_pose, object_location):
        """
        Compute the robot base final heading angle such that the robot faces the object as it approaches.
        Args:
            base_pose: [x, y, theta] list or array (robot base position and current heading)
            object_location: [x, y, z] list or array (object position)
        Returns:
            heading (float): angle in radians, so the robot faces the object
        """
        dx = object_location[0] - base_pose[0]
        dy = object_location[1] - base_pose[1]
        heading = math.atan2(dy, dx)
        return heading 

    def find_base_waypoint(self, command_waypoints, target_ee_pos):
        """
        Find the optimal base waypoint position that places the base exactly 
        FIXED_END_EFFECTOR_OFFSET distance away from the target end effector position.
        
        Args:
            command_waypoints: List of waypoints from the original command
            target_ee_pos: Target end effector position [x, y]
            
        Returns:
            List of waypoints for base movement
        """
        new_waypoint = None  # Find new_waypoint such that distance(new_waypoint, target_ee_pos) == FIXED_END_EFFECTOR_OFFSET
        reversed_waypoints = command_waypoints[::-1]
        
        # Try to find intersection point along the path that is exactly FIXED_END_EFFECTOR_OFFSET away
        for idx in range(1, len(reversed_waypoints)):
            start = reversed_waypoints[idx - 1]
            end = reversed_waypoints[idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - target_ee_pos[0], start[1] - target_ee_pos[1])
            t2 = intersect(d, f, self.FIXED_END_EFFECTOR_OFFSET)
            if t2 is not None:
                new_waypoint = (start[0] + t2 * d[0], start[1] + t2 * d[1])
                break
                
        if new_waypoint is not None:
            # Discard all waypoints that are too close to target_ee_pos
            waypoints = reversed_waypoints[idx:][::-1] + [new_waypoint]
        else:
            # Base is too close to target end effector position and needs to back up
            print('Warning: Base needs to deviate from commanded path to reach target position, watch out for potential collisions')
            curr_position = command_waypoints[0]
            signed_dist = distance(curr_position, target_ee_pos) - self.FIXED_END_EFFECTOR_OFFSET
            dx = target_ee_pos[0] - curr_position[0]
            dy = target_ee_pos[1] - curr_position[1]
            target_heading = restrict_heading_range(math.atan2(dy, dx))
            target_position = (curr_position[0] + signed_dist * math.cos(target_heading), curr_position[1] + signed_dist * math.sin(target_heading))
            waypoints = [curr_position, target_position]
            
        return waypoints

    def build_base_command(self, command):
        # Use the dedicated function to find optimal base waypoints
        target_ee_pos = command['waypoints'][-1]
        waypoints = self.find_base_waypoint(command['waypoints'], target_ee_pos)
            
        return {'waypoints': waypoints, 
                'target_ee_pos': target_ee_pos} 

    def detect_objects_from_ground_truth(self, obs):
        """Detect objects using ground truth from MuJoCo simulation and find the one with smallest x value"""
        detected_objects = []
        
        # Get all three cube positions from MuJoCo environment
        cubes = []
        for i in range(1, 4):
            cube_key = f'cube{i}_pos'
            if cube_key in obs:
                cube_pos = obs[cube_key].copy()
                cubes.append((cube_pos, i))
                print(f"Detected cube {i} at position: {cube_pos}")
            else:
                print(f"Warning: {cube_key} not found in observation")
        
        if cubes:
            # Sort cubes by x position and select the one with smallest x value
            cubes.sort(key=lambda x: x[0][0])  # Sort by x coordinate (first element of position)
            target_cube_pos, target_cube_id = cubes[0]
            detected_objects.append(target_cube_pos)
            print(f"Selected cube {target_cube_id} with smallest x value: {target_cube_pos[0]:.3f}")
        
        return detected_objects 

