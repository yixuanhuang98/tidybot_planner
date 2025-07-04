import math
from typing import Dict, Any, Optional
import numpy as np

from agent.base_agent import BaseAgent


class GoToCabinetHandlePolicyRight(BaseAgent):
    """State machine for right cabinet handle: move base to approach pose, move arm to handle, grasp, move base away, hold"""
    
    def __init__(self):
        super().__init__()
        
        # Target handle pose and orientation (right door)
        self.target_pos = np.array([0.652, -0.078, 0])
        self.target_quat = np.array([0.5, 0.5, 0.5, 0.5])  # Placeholder quaternion for right handle
        self.gripper_open = np.array([0.0])
        self.gripper_closed = np.array([0.6])
        
        # State machine
        self.state = 'move_base'
        self.initial_base = None
        self.arm_target = self.target_pos
        self.quat_target = self.target_quat
        
        # Parameters
        self.ee_offset = 0.75  # End effector offset
        self.approach_offset = 0.10  # Approach offset (meters)
        
        # Base following parameters
        self.LOOKAHEAD_DISTANCE = 0.3  # 30 cm
        self.position_tolerance = 0.005  # 0.5 cm
        self.heading_tolerance = math.radians(2.1)  # 2.1 degrees
        
        # Base movement state
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.lookahead_position = None
        self.current_command = None
        self.base_target_away = None

    def reset(self):
        """Reset the policy state"""
        self.state = 'move_base'
        self.initial_base = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.lookahead_position = None
        self.current_command = None
        self.base_target_away = None

    def step(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute right cabinet handle manipulation"""
        base_pose = obs['base_pose']
        arm_pos = obs['arm_pos']
        arm_quat = obs['arm_quat']
        gripper_pos = obs['gripper_pos']

        # Update target position if rightdoor_pos is available
        if 'rightdoor_pos' in obs:
            self.target_pos = obs['rightdoor_pos'].copy()
        else:
            print("[GoToCabinetHandlePolicyRight] WARNING: rightdoor_pos not found, using placeholder")

        # Initialize base approach on first call
        if self.initial_base is None:
            self._initialize_base_approach(base_pose)

        # State machine execution
        if self.state == 'move_base':
            return self._handle_move_base(obs, base_pose, arm_pos, arm_quat)
        elif self.state == 'move_arm':
            return self._handle_move_arm(base_pose, arm_pos, arm_quat, gripper_pos)
        elif self.state == 'grasp':
            return self._handle_grasp(base_pose, arm_pos, arm_quat, gripper_pos)
        elif self.state == 'move_base_away':
            return self._handle_move_base_away(base_pose, arm_pos, arm_quat, gripper_pos)
        else:  # done state
            return self._handle_done(base_pose, arm_pos, arm_quat, gripper_pos)

    def _initialize_base_approach(self, base_pose: np.ndarray):
        """Initialize base approach planning"""
        self.initial_base = base_pose.copy()
        
        # Create handle approach command
        handle_command = {
            'primitive_name': 'pick',
            'waypoints': [base_pose[:2].tolist(), self.target_pos[:2].tolist()],
            'handle_3d_pos': self.target_pos.copy()
        }
        
        print(f"[GoToCabinetHandlePolicyRight] Right handle at: {self.target_pos}")
        
        # Build base command
        base_command = self._build_base_command(handle_command)
        if base_command:
            self.current_command = handle_command
            self.base_waypoints = base_command['waypoints']
            self.target_ee_pos = base_command['target_ee_pos']
            self.current_waypoint_idx = 1
            self.lookahead_position = None
            
            # Compute away position
            if len(self.base_waypoints) > 0:
                final_pos = self.base_waypoints[-1]
                theta = base_pose[2]
                approach_dir = np.array([np.cos(theta), np.sin(theta)])
                self.base_target_away = np.zeros(3)
                self.base_target_away[:2] = np.array(final_pos) - 0.1 * approach_dir
                self.base_target_away[2] = theta
        else:
            print("[GoToCabinetHandlePolicyRight] Failed to build base command")
            # Simple fallback
            handle_xy = self.target_pos[:2]
            theta = base_pose[2]
            approach_dir = np.array([np.cos(theta), np.sin(theta)])
            base_target = np.zeros(3)
            base_target[:2] = handle_xy - self.ee_offset * approach_dir
            base_target[2] = theta
            self.base_waypoints = [base_pose[:2].tolist(), base_target[:2].tolist()]
            self.current_waypoint_idx = 1
            self.base_target_away = base_target.copy()
            self.base_target_away[:2] -= 0.1 * approach_dir

    def _handle_move_base(self, obs: Dict[str, Any], base_pose: np.ndarray,
                         arm_pos: np.ndarray, arm_quat: np.ndarray) -> Dict[str, Any]:
        """Handle base movement state"""
        action = self._execute_base_movement(obs)
        if action is None:  # Base movement complete
            print("[GoToCabinetHandlePolicyRight] Base movement complete!")
            
            # Calculate arm target relative to new base pose
            handle_world = self.target_pos
            base_xy = base_pose[:2]
            base_theta = base_pose[2]
            dx = handle_world[0] - base_xy[0]
            dy = handle_world[1] - base_xy[1]
            dist = np.linalg.norm([dx, dy])
            
            # Transform to base frame
            c, s = np.cos(-base_theta), np.sin(-base_theta)
            rel_x = c * dx - s * dy
            rel_y = s * dx + c * dy
            rel_z = handle_world[2]
            
            # Apply approach offset
            if dist > 1e-6:
                approach_dir = np.array([rel_x, rel_y]) / dist
            else:
                approach_dir = np.array([1.0, 0.0])
            
            offset_xy = np.array([rel_x, rel_y]) - self.approach_offset * approach_dir
            self.arm_target = np.array([offset_xy[0], offset_xy[1], rel_z])
            self.state = 'move_arm'
        else:
            # Hold arm at current position during base movement
            action['arm_pos'] = arm_pos.copy()
            action['arm_quat'] = arm_quat.copy()
            action['gripper_pos'] = self.gripper_open.copy()
        
        return action

    def _handle_move_arm(self, base_pose: np.ndarray, arm_pos: np.ndarray,
                        arm_quat: np.ndarray, gripper_pos: np.ndarray) -> Dict[str, Any]:
        """Handle arm movement to right handle"""
        action = self.create_action(
            base_pose=base_pose.copy(),
            arm_pos=self.arm_target.copy(),
            arm_quat=self.quat_target.copy(),
            gripper_pos=self.gripper_open.copy()
        )
        
        # Check if arm reached target
        pos_close = self.check_position_reached(arm_pos, self.arm_target, tolerance=0.01)
        quat_close = self.check_orientation_reached(arm_quat, self.quat_target, angle_tolerance=np.deg2rad(5))
        
        if pos_close and quat_close:
            print("[GoToCabinetHandlePolicyRight] Arm at right handle, starting grasp")
            self.state = 'grasp'
        
        return action

    def _handle_grasp(self, base_pose: np.ndarray, arm_pos: np.ndarray,
                     arm_quat: np.ndarray, gripper_pos: np.ndarray) -> Dict[str, Any]:
        """Handle grasping the right handle"""
        action = self.create_action(
            base_pose=base_pose.copy(),
            arm_pos=self.arm_target.copy(),
            arm_quat=self.quat_target.copy(),
            gripper_pos=self.gripper_closed.copy()
        )
        
        if np.allclose(gripper_pos, self.gripper_closed, atol=0.1):
            print("[GoToCabinetHandlePolicyRight] Right handle grasped, moving base away")
            self.state = 'move_base_away'
        
        return action

    def _handle_move_base_away(self, base_pose: np.ndarray, arm_pos: np.ndarray,
                              arm_quat: np.ndarray, gripper_pos: np.ndarray) -> Dict[str, Any]:
        """Handle moving base away while holding right handle"""
        action = self.create_action(
            base_pose=self.base_target_away.copy(),
            arm_pos=self.arm_target.copy(),
            arm_quat=self.quat_target.copy(),
            gripper_pos=self.gripper_closed.copy()
        )
        
        if self.check_position_reached(base_pose, self.base_target_away, tolerance=0.01):
            print("[GoToCabinetHandlePolicyRight] Base moved away with right handle, task complete")
            self.state = 'done'
        
        return action

    def _handle_done(self, base_pose: np.ndarray, arm_pos: np.ndarray,
                    arm_quat: np.ndarray, gripper_pos: np.ndarray) -> Dict[str, Any]:
        """Handle done state - hold position"""
        return self.create_action(
            base_pose=self.base_target_away.copy(),
            arm_pos=self.arm_target.copy(),
            arm_quat=self.quat_target.copy(),
            gripper_pos=self.gripper_closed.copy()
        )

    def _execute_base_movement(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute base movement following waypoints like MotionPlannerPolicy"""
        base_pose = obs['base_pose']
        
        # Check if we've reached the final waypoint
        if self.current_waypoint_idx >= len(self.base_waypoints):
            print("[GoToCabinetHandlePolicyRight] All waypoints completed")
            return None  # Movement complete
        
        print(f"[GoToCabinetHandlePolicyRight] Current waypoint index: {self.current_waypoint_idx}/{len(self.base_waypoints)}")
        print(f"[GoToCabinetHandlePolicyRight] Base waypoints: {self.base_waypoints}")
        
        # Compute lookahead position (simplified version of BaseController logic)
        while True:
            if self.current_waypoint_idx >= len(self.base_waypoints):
                self.lookahead_position = None
                print("[GoToCabinetHandlePolicyRight] Reached end of waypoints, no lookahead")
                break
                
            start = self.base_waypoints[self.current_waypoint_idx - 1]
            end = self.base_waypoints[self.current_waypoint_idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - base_pose[0], start[1] - base_pose[1])
            t2 = self.intersect(d, f, self.LOOKAHEAD_DISTANCE)
            
            print(f"[GoToCabinetHandlePolicyRight] Waypoint {self.current_waypoint_idx}: start={start}, end={end}")
            print(f"[GoToCabinetHandlePolicyRight] d={d}, f={f}, t2={t2}")
            
            if t2 is not None:
                self.lookahead_position = [start[0] + t2 * d[0], start[1] + t2 * d[1]]
                print(f"[GoToCabinetHandlePolicyRight] Lookahead position: {self.lookahead_position}")
                break
            if self.current_waypoint_idx == len(self.base_waypoints) - 1:
                self.lookahead_position = None
                print("[GoToCabinetHandlePolicyRight] Last waypoint, no lookahead")
                break
            print(f"[GoToCabinetHandlePolicyRight] Moving to next waypoint: {self.current_waypoint_idx + 1}")
            self.current_waypoint_idx += 1
        
        # Determine target position
        if self.lookahead_position is None:
            target_position = self.base_waypoints[-1]
            print(f"[GoToCabinetHandlePolicyRight] Using final waypoint as target: {target_position}")
            # Check if we've reached the final position
            position_error = self.distance_2d(base_pose[:2], target_position)
            print(f"[GoToCabinetHandlePolicyRight] Position error to final target: {position_error:.3f} (tolerance: {self.position_tolerance})")
            if position_error < self.position_tolerance:
                print("[GoToCabinetHandlePolicyRight] Reached final position within tolerance")
                return None  # Movement complete
        else:
            target_position = self.lookahead_position
            print(f"[GoToCabinetHandlePolicyRight] Using lookahead as target: {target_position}")
        
        # Compute target heading
        target_heading = base_pose[2]
        if self.target_ee_pos is not None:
            # Turn to face target end effector position
            dx = self.target_ee_pos[0] - base_pose[0]
            dy = self.target_ee_pos[1] - base_pose[1]
            desired_heading = math.atan2(dy, dx)
            
            print(f"[GoToCabinetHandlePolicyRight] Target EE: {self.target_ee_pos}, dx={dx:.3f}, dy={dy:.3f}, desired_heading={desired_heading:.3f}")
            
            frac = 1
            if self.lookahead_position is not None:
                # Turn slowly at first, more quickly as we approach
                remaining_path_length = self.LOOKAHEAD_DISTANCE
                curr_waypoint = self.lookahead_position
                for idx in range(self.current_waypoint_idx, len(self.base_waypoints)):
                    next_waypoint = self.base_waypoints[idx]
                    remaining_path_length += self.distance_2d(curr_waypoint, next_waypoint)
                    curr_waypoint = next_waypoint
                frac = math.sqrt(self.LOOKAHEAD_DISTANCE / max(remaining_path_length, self.LOOKAHEAD_DISTANCE))
            
            heading_diff = self.normalize_angle(desired_heading - base_pose[2])
            target_heading += frac * heading_diff
            print(f"[GoToCabinetHandlePolicyRight] Heading: current={base_pose[2]:.3f}, desired={desired_heading:.3f}, diff={heading_diff:.3f}, frac={frac:.3f}, target={target_heading:.3f}")
        
        # Create action to move towards target
        action = {
            'base_pose': np.array([target_position[0], target_position[1], target_heading]),
            'arm_pos': obs['arm_pos'].copy(),
            'arm_quat': obs['arm_quat'].copy(),
            'gripper_pos': obs['gripper_pos'].copy(),
        }
        
        return action

    def _build_base_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build base command using exact logic from MotionPlannerPolicy"""
        assert command['primitive_name'] in {'move', 'pick', 'place', 'toss', 'shelf', 'drawer'}

        # Base movement only
        if command['primitive_name'] == 'move':
            return {'waypoints': command['waypoints'], 'target_ee_pos': None, 'position_tolerance': 0.1}

        # Modify waypoints so that the end effector is placed at the target end effector position
        target_ee_pos = command['waypoints'][-1]
        end_effector_offset = self.get_end_effector_offset(command['primitive_name'])
        new_waypoint = None  # Find new_waypoint such that distance(new_waypoint, target_ee_pos) == end_effector_offset
        reversed_waypoints = command['waypoints'][::-1]
        
        for idx in range(1, len(reversed_waypoints)):
            start = reversed_waypoints[idx - 1]
            end = reversed_waypoints[idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - target_ee_pos[0], start[1] - target_ee_pos[1])
            t2 = self.intersect(d, f, end_effector_offset)
            if t2 is not None:
                new_waypoint = (start[0] + t2 * d[0], start[1] + t2 * d[1])
                break
                
        if new_waypoint is not None:
            # Discard all waypoints that are too close to target_ee_pos
            waypoints = reversed_waypoints[idx:][::-1] + [new_waypoint]
        else:
            # Base is too close to target end effector position and needs to back up
            print('[GoToCabinetHandlePolicyRight] Warning: Base needs to deviate from commanded path to reach target position, watch out for potential collisions')
            curr_position = command['waypoints'][0]
            signed_dist = self.distance_2d(curr_position, target_ee_pos) - end_effector_offset
            dx = target_ee_pos[0] - curr_position[0]
            dy = target_ee_pos[1] - curr_position[1]
            target_heading = self.normalize_angle(math.atan2(dy, dx))
            target_position = (curr_position[0] + signed_dist * math.cos(target_heading), curr_position[1] + signed_dist * math.sin(target_heading))
            waypoints = [curr_position, target_position]
            
        return {'waypoints': waypoints, 'target_ee_pos': target_ee_pos}