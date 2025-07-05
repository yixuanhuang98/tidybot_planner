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

    # Base stopping verification parameters
    BASE_STOP_VELOCITY_THRESHOLD = 0.01  # Consider base stopped if velocity < 1cm/s
    BASE_STOP_CONFIRMATION_FRAMES = 5    # Confirm base has stopped for 5 consecutive frames

    # Grasping parameters
    GRASP_SUCCESS_THRESHOLD = 0.6
    GRASP_PROGRESS_THRESHOLD = 0.3
    GRASP_TIMEOUT_S = 3.0
    PLACE_SUCCESS_THRESHOLD = 0.2

    def __init__(self):
        # Motion planning state - following controller.py pattern
        self.state = 'idle'  # States: idle, moving, base_stopping, manipulating
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.grasp_state = None
        
        # Base following parameters
        self.lookahead_position = None
        
        # Base stopping verification
        self.prev_base_poses = []  # Store recent base poses to check if stopped
        self.base_stop_frames = 0  # Counter for consecutive stopped frames
        
        # Object and target locations (using ground truth from MuJoCo)
        self.object_location = None
        self.target_location = None
        
        # Enable policy execution immediately (no web interface required)
        self.enabled = True
        self.episode_ended = False
        
        print(f'MMMP policy initialized - base will position at exactly {self.FIXED_END_EFFECTOR_OFFSET}m from target')

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
        
        # Reset base stopping verification
        self.prev_base_poses = []
        self.base_stop_frames = 0
        
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

        # State machine with strict base-arm sequencing
        if self.state == 'idle':
            # Detect objects and plan new command
            detected_objects = self.detect_objects_from_ground_truth(obs)
            if not detected_objects:
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

            # waypoints are [x, y, theta]
            # waypoints[0] is the current base pose
            # waypoints[1,:2]  = self.find_base_waypoint(command['waypoints'], target_ee_pos)
            # waypoints[1,2] = self.find_base_heading(command['waypoints'][0], self.object_location)
            
            

            print(f"Base command: {base_command}")
            
            if base_command:
                self.current_command = pick_command
                self.base_waypoints = base_command['waypoints']
                self.target_ee_pos = base_command['target_ee_pos']
                self.current_waypoint_idx = 1
                self.lookahead_position = None
                self.state = 'moving'
                self.prev_base_poses = []  # Reset base pose tracking
                self.base_stop_frames = 0
                print(f"Starting base movement to object at {self.object_location}")
                print(f"Base waypoints: {self.base_waypoints}")
                print(f"Target EE position: {self.target_ee_pos}")
                print(f"Will position base at exactly {self.FIXED_END_EFFECTOR_OFFSET}m from target")
            else:
                print("Failed to build base command")

                
        elif self.state == 'moving':
            # Execute base movement following waypoints
            action = self.execute_base_movement(obs)
            if action is None:  # Base movement complete
                print("Base movement complete! Verifying base has stopped...")
                # Transition to base_stopping state to confirm base has stopped
                self.state = 'base_stopping'
                self.prev_base_poses = []
                self.base_stop_frames = 0
            else:
                # Print target base pose from action
                if action and 'base_pose' in action:
                    target_pose = action['base_pose']
                    print(f"Target base pose: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
            return action
            
        elif self.state == 'base_stopping':
            # Verify that base has completely stopped before starting arm manipulation
            current_base_pose = base_pose[:2]  # Only x, y for position tracking
            
            # Store recent base poses
            self.prev_base_poses.append(current_base_pose.copy())
            if len(self.prev_base_poses) > self.BASE_STOP_CONFIRMATION_FRAMES:
                self.prev_base_poses.pop(0)
            
            # Check if base has stopped moving
            base_stopped = False
            if len(self.prev_base_poses) >= self.BASE_STOP_CONFIRMATION_FRAMES:
                # Calculate velocity over recent frames
                max_velocity = 0
                for i in range(1, len(self.prev_base_poses)):
                    velocity = distance(self.prev_base_poses[i-1], self.prev_base_poses[i])
                    max_velocity = max(max_velocity, velocity)
                
                if max_velocity < self.BASE_STOP_VELOCITY_THRESHOLD:
                    self.base_stop_frames += 1
                    print(f"Base stopping verification: frame {self.base_stop_frames}/{self.BASE_STOP_CONFIRMATION_FRAMES}, max_vel: {max_velocity:.4f}")
                else:
                    self.base_stop_frames = 0  # Reset if base is still moving
                    print(f"Base still moving, max velocity: {max_velocity:.4f}")
                
                if self.base_stop_frames >= self.BASE_STOP_CONFIRMATION_FRAMES:
                    base_stopped = True
            
            if base_stopped:
                # Verify we're at correct distance from target
                if self.target_ee_pos is not None:
                    distance_to_target = distance(base_pose[:2], self.target_ee_pos)
                    distance_error = abs(distance_to_target - self.FIXED_END_EFFECTOR_OFFSET)
                    print(f"Base stopped! Distance to target: {distance_to_target:.3f}m, target: {self.FIXED_END_EFFECTOR_OFFSET}m, error: {distance_error:.4f}m")
                    
                    if self.current_command['primitive_name'] == 'pick':
                        tolerance = self.GRASP_BASE_TOLERANCE
                    else:
                        tolerance = self.PLACE_BASE_TOLERANCE
                        
                    if distance_error < tolerance:
                        self.state = 'manipulating'
                        print("Base properly positioned and stopped! Starting arm manipulation")
                    else:
                        print(f'Base stopped but not at correct distance (error: {distance_error*100:.1f}cm > {tolerance*100:.1f}cm)')
                        self.state = 'idle'
                else:
                    self.state = 'idle'
            
            # Hold current pose while verifying base has stopped
            action = {
                'base_pose': base_pose.copy(),
                'arm_pos': arm_pos.copy(),
                'arm_quat': arm_quat.copy(),
                'gripper_pos': gripper_pos.copy(),
            }
            return action
            
        elif self.state == 'manipulating':
            # Execute arm manipulation (same logic as original)
            if self.current_command['primitive_name'] == 'pick':
                if self.grasp_state is None:
                    self.grasp_state = PickState.APPROACH

                # Define default targets to hold current pose
                target_arm_pos = arm_pos.copy()
                target_arm_quat = arm_quat.copy()
                target_gripper_pos = gripper_pos.copy()

                # Position arm above object and close gripper to grasp
                object_3d_pos = self.current_command['object_3d_pos']
                # Calculate global position difference with better approach height
                global_diff = np.array([
                    object_3d_pos[0] - base_pose[0],
                    object_3d_pos[1] - base_pose[1], 
                    object_3d_pos[2] + self.PICK_APPROACH_HEIGHT_OFFSET - self.ROBOT_BASE_HEIGHT
                ])
                
                # Transform to base's local coordinate frame (account for base rotation)
                base_angle = base_pose[2]
                cos_angle = math.cos(-base_angle)  # Negative for inverse rotation
                sin_angle = math.sin(-base_angle)
                
                object_relative_pos = np.array([
                    cos_angle * global_diff[0] - sin_angle * global_diff[1],
                    sin_angle * global_diff[0] + cos_angle * global_diff[1],
                    global_diff[2]  # Z component unchanged
                ])
                print(f"Arm manipulation: global_diff = {global_diff}")
                
                print(f"Base angle: {base_pose[2]:.3f} rad ({math.degrees(base_pose[2]):.1f} deg)")
                print(f"Object global pos: {self.current_command.get('object_3d_pos', 'N/A')}")
                
                print(f"Current arm pos: {arm_pos}")
                print(f"Position error: {np.linalg.norm(arm_pos - object_relative_pos):.4f}m")
                print(f"Grasp state: {self.grasp_state}")
                
                if self.grasp_state == PickState.APPROACH:
                    # Step 1: Position arm well above object with open gripper (safe approach)
                    target_arm_pos = object_relative_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([0.0])  # Gripper open

                    print(f"Step 1: Positioning arm above object with open gripper")
                    if np.allclose(arm_pos, target_arm_pos, atol=0.03):  # Tighter tolerance: 3cm
                        self.grasp_state = PickState.LOWER
                        print("Arm positioned above object, moving to lower approach")
                
                elif self.grasp_state == PickState.LOWER:
                    # Step 2: Lower gripper closer to object for precise grasping
                    target_arm_pos = object_relative_pos.copy()
                    target_arm_pos[2] -= self.PICK_LOWER_DIST  # Lower by 8cm for closer approach
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([0.0])  # Gripper open

                    print(f"Step 2: Lowering gripper for precise approach... target: {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}")
                    if np.allclose(arm_pos, target_arm_pos, atol=0.02):  # Very tight tolerance: 2cm
                        self.grasp_state = PickState.GRASP
                        print("Gripper lowered to grasping position, closing gripper")
                
                elif self.grasp_state == PickState.GRASP:
                    # Step 3: Close gripper to grasp
                    target_arm_pos = object_relative_pos.copy()
                    target_arm_pos[2] -= self.PICK_LOWER_DIST  # Maintain lowered position
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([1.0])  # Close gripper

                    print(f"Step 3: Closing gripper... current position: {gripper_pos[0]:.3f}")
                    
                    # Initialize grasp attempt tracking
                    if not hasattr(self, 'grasp_start_time'):
                        self.grasp_start_time = time.time()
                        self.initial_gripper_pos = gripper_pos[0]
                        print(f"Started grasp attempt, initial gripper pos: {self.initial_gripper_pos:.3f}")
                    
                    # Check for successful grasp (multiple criteria)
                    gripper_closed_enough = gripper_pos[0] > self.GRASP_SUCCESS_THRESHOLD
                    gripper_progress = (gripper_pos[0] - self.initial_gripper_pos) > self.GRASP_PROGRESS_THRESHOLD
                    grasp_timeout = (time.time() - self.grasp_start_time) > self.GRASP_TIMEOUT_S
                    
                    if gripper_closed_enough or gripper_progress or grasp_timeout:
                        if gripper_closed_enough or gripper_progress:
                            print(f"Grasp successful! Gripper pos: {gripper_pos[0]:.3f}, progress: {gripper_pos[0] - self.initial_gripper_pos:.3f}")
                        else:
                            print(f"Grasp timeout reached, proceeding with current grip: {gripper_pos[0]:.3f}")
                        
                        self.grasp_state = PickState.LIFT
                        # Clean up tracking variables
                        delattr(self, 'grasp_start_time')
                        delattr(self, 'initial_gripper_pos')
                        print("Moving to lift phase!")
                
                elif self.grasp_state == PickState.LIFT:
                    # Step 4: Lift object from the grasping position
                    lifted_pos = object_relative_pos.copy()
                    lifted_pos[2] += (self.PICK_LIFT_DIST - self.PICK_LOWER_DIST) # Net lift
                    target_arm_pos = lifted_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([1.0])  # Gripper closed

                    print(f"Step 4: Lifting object... target height: {target_arm_pos[2]:.3f}, current: {arm_pos[2]:.3f}")
                    if np.allclose(arm_pos, target_arm_pos, atol=0.05):  # 5cm tolerance
                        print("Object lifted successfully! Now moving to placement location.")
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
                            self.prev_base_poses = []  # Reset base tracking
                            self.base_stop_frames = 0
                            print(f"Starting base movement to placement location at {self.target_location}")
                        else:
                            print("Failed to build place command")
                            self.episode_ended = True
                            self.state = 'idle'

                # Create action from targets (hold base pose steady during arm manipulation)
                action = {
                    'base_pose': base_pose.copy(),  # Keep base stationary
                    'arm_pos': target_arm_pos,
                    'arm_quat': target_arm_quat,
                    'gripper_pos': target_gripper_pos,
                }
                return action

            elif self.current_command['primitive_name'] == 'place':
                if self.grasp_state is None:
                    self.grasp_state = PlaceState.APPROACH

                # Define default targets to hold current pose
                target_arm_pos = arm_pos.copy()
                target_arm_quat = arm_quat.copy()
                target_gripper_pos = gripper_pos.copy()

                # Position arm above placement location and open gripper
                target_3d_pos = self.current_command['target_3d_pos']
                # Calculate global position difference
                global_diff = np.array([
                    target_3d_pos[0] - base_pose[0],
                    target_3d_pos[1] - base_pose[1], 
                    target_3d_pos[2] + self.PLACE_APPROACH_HEIGHT_OFFSET - self.ROBOT_BASE_HEIGHT
                ])
                
                # Transform to base's local coordinate frame (account for base rotation)
                base_angle = base_pose[2]
                cos_angle = math.cos(-base_angle)  # Negative for inverse rotation
                sin_angle = math.sin(-base_angle)
                
                target_relative_pos = np.array([
                    cos_angle * global_diff[0] - sin_angle * global_diff[1],
                    sin_angle * global_diff[0] + cos_angle * global_diff[1],
                    global_diff[2]  # Z component unchanged
                ])
                
                print(f"Placing: target_relative_pos = {target_relative_pos}")
                print(f"Target EE pos: {self.target_ee_pos}, Base pose: {base_pose}")
                print(f"Grasp state: {self.grasp_state}")
                
                if self.grasp_state == PlaceState.APPROACH:
                    # Step 1: Position arm above placement location with closed gripper
                    target_arm_pos = target_relative_pos
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([1.0])  # Gripper closed

                    print(f"Step 1: Positioning arm above placement location with closed gripper")
                    if np.allclose(arm_pos, target_arm_pos, atol=0.05):  # 5cm tolerance
                        self.grasp_state = PlaceState.RELEASE
                        print("Arm positioned above placement location, opening gripper")
                
                elif self.grasp_state == PlaceState.RELEASE:
                    # Step 2: Open gripper to place object
                    target_arm_pos = target_relative_pos  # Maintain arm position
                    target_arm_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Gripper down
                    target_gripper_pos = np.array([0.0])  # Gripper open

                    print(f"Step 2: Opening gripper... current position: {gripper_pos[0]:.3f}")
                    if gripper_pos[0] < self.PLACE_SUCCESS_THRESHOLD:
                        print("Object placed successfully! Task complete.")
                        self.episode_ended = True  # End the episode
                        self.state = 'idle'
                
                # Create action from targets (hold base pose steady during arm manipulation)
                action = {
                    'base_pose': base_pose.copy(),  # Keep base stationary
                    'arm_pos': target_arm_pos,
                    'arm_quat': target_arm_quat,
                    'gripper_pos': target_gripper_pos,
                }
                return action

        # Default: hold current pose
        action = {
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos.copy(),
            'arm_quat': arm_quat.copy(),
            'gripper_pos': gripper_pos.copy(),
        }
        print(f"Default action - holding current pose")
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

