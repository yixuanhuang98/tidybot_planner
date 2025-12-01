# Author: Jimmy Wu
# Date: October 2024

import logging
import math
import socket
import threading
import time
from queue import Queue
import cv2 as cv
import numpy as np
import zmq
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from scipy.spatial.transform import Rotation as R
from constants import POLICY_SERVER_HOST, POLICY_SERVER_PORT, POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT

class Policy:
    """Abstract base class for robot policies."""
    
    def reset(self):
        """Reset the policy for a new episode."""
        raise NotImplementedError

    def step(self, obs):
        """Execute one policy step given an observation.
        
        Args:
            obs: Dictionary containing robot state observations
            
        Returns:
            Action dictionary or None/end_episode signal
        """
        raise NotImplementedError

class WebServer:
    """Flask web server for serving WebXR phone interface and handling teleop messages."""
    
    def __init__(self, queue):
        """Initialize the web server with a message queue.
        
        Args:
            queue: Queue object for receiving messages from WebXR client
        """
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.queue = queue

        @self.app.route('/')
        def index():
            """Serve the main HTML page for the WebXR interface."""
            return render_template('index.html')

        @self.socketio.on('message')
        def handle_message(data):
            """Handle incoming WebSocket messages from WebXR client.
            
            Echoes timestamp back for RTT calculation and adds message to queue.
            
            Args:
                data: Dictionary containing teleop data (position, orientation, etc.)
            """
            # Send the timestamp back for RTT calculation (expected RTT on 5 GHz Wi-Fi is 7 ms)
            emit('echo', data['timestamp'])

            # Add data to queue for processing
            self.queue.put(data)

        # Reduce verbose Flask log output
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    def run(self):
        """Start the Flask web server on all network interfaces."""
        # Get IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 1))
            address = s.getsockname()[0]
        except Exception:
            address = '127.0.0.1'
        finally:
            s.close()
        print(f'Starting server at {address}:5000')
        self.socketio.run(self.app, host='0.0.0.0')

DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])  # iPhone 14 Pro

# Convert coordinate system from WebXR to robot
def convert_webxr_pose(pos, quat):
    """Convert pose from WebXR coordinate system to robot coordinate system.
    
    WebXR: +x right, +y up, +z back
    Robot: +x forward, +y left, +z up
    
    Also applies device camera offset so rotations are around device center.
    
    Args:
        pos: Dictionary with 'x', 'y', 'z' position in WebXR frame
        quat: Dictionary with 'x', 'y', 'z', 'w' quaternion in WebXR frame
        
    Returns:
        tuple: (position_array, Rotation_object) in robot coordinate frame
    """
    # WebXR: +x right, +y up, +z back; Robot: +x forward, +y left, +z up
    pos = np.array([-pos['z'], -pos['x'], pos['y']], dtype=np.float64)
    rot = R.from_quat([-quat['z'], -quat['x'], quat['y'], quat['w']])

    # Apply offset so that rotations are around device center instead of device camera
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)

    return pos, rot

TWO_PI = 2 * math.pi

class TeleopController:
    """Controller for processing WebXR teleoperation commands and generating robot actions."""
    
    def __init__(self):
        """Initialize teleop controller with empty state."""
        # Teleop device IDs
        self.primary_device_id = None    # Primary device controls either the arm or the base
        self.secondary_device_id = None  # Optional secondary device controls the base
        self.enabled_counts = {}

        # Mobile base pose
        self.base_pose = None

        # Teleop targets
        self.targets_initialized = False
        self.base_target_pose = None
        self.arm_target_pos = None
        self.arm_target_rot = None
        self.gripper_target_pos = None

        # WebXR reference poses
        self.base_xr_ref_pos = None
        self.base_xr_ref_rot_inv = None
        self.arm_xr_ref_pos = None
        self.arm_xr_ref_rot_inv = None

        # Robot reference poses
        self.base_ref_pose = None
        self.arm_ref_pos = None
        self.arm_ref_rot = None
        self.arm_ref_base_pose = None  # For optional secondary control of base
        self.gripper_ref_pos = None

    def process_message(self, data):
        """Process incoming WebXR teleop message and update target poses.
        
        Handles device assignment (primary/secondary), reference pose initialization,
        and computes target poses for base, arm, and gripper based on WebXR input.
        
        Args:
            data: Dictionary containing WebXR pose data and teleop mode
        """
        if not self.targets_initialized:
            return

        # Use device ID to disambiguate between primary and secondary devices
        device_id = data['device_id']

        # Update enabled count for the device that sent this message
        self.enabled_counts[device_id] = self.enabled_counts.get(device_id, 0) + 1 if 'teleop_mode' in data else 0

        # Assign primary and secondary devices
        if self.enabled_counts[device_id] > 2:
            if self.primary_device_id is None and device_id != self.secondary_device_id:
                # Note: We skip the first 2 steps because WebXR pose updates have higher latency than touch events
                self.primary_device_id = device_id
            elif self.secondary_device_id is None and device_id != self.primary_device_id:
                self.secondary_device_id = device_id
        elif self.enabled_counts[device_id] == 0:
            if device_id == self.primary_device_id:
                self.primary_device_id = None  # Primary device no longer enabled
                self.base_xr_ref_pos = None
                self.arm_xr_ref_pos = None
            elif device_id == self.secondary_device_id:
                self.secondary_device_id = None
                self.base_xr_ref_pos = None

        # Teleop is enabled
        if self.primary_device_id is not None and 'teleop_mode' in data:
            pos, rot = convert_webxr_pose(data['position'], data['orientation'])

            # Base movement
            if data['teleop_mode'] == 'base' or device_id == self.secondary_device_id:  # Note: Secondary device can only control base
                # Store reference poses
                if self.base_xr_ref_pos is None:
                    self.base_ref_pose = self.base_pose.copy()
                    self.base_xr_ref_pos = pos[:2]
                    self.base_xr_ref_rot_inv = rot.inv()

                # Position
                self.base_target_pose[:2] = self.base_ref_pose[:2] + (pos[:2] - self.base_xr_ref_pos)

                # Orientation
                base_fwd_vec_rotated = (rot * self.base_xr_ref_rot_inv).apply([1.0, 0.0, 0.0])
                base_target_theta = self.base_ref_pose[2] + math.atan2(base_fwd_vec_rotated[1], base_fwd_vec_rotated[0])
                self.base_target_pose[2] += (base_target_theta - self.base_target_pose[2] + math.pi) % TWO_PI - math.pi  # Unwrapped

            # Arm movement
            elif data['teleop_mode'] == 'arm':
                # Store reference poses
                if self.arm_xr_ref_pos is None:
                    self.arm_xr_ref_pos = pos
                    self.arm_xr_ref_rot_inv = rot.inv()
                    self.arm_ref_pos = self.arm_target_pos.copy()
                    self.arm_ref_rot = self.arm_target_rot
                    self.arm_ref_base_pose = self.base_pose.copy()
                    self.gripper_ref_pos = self.gripper_target_pos

                # Rotations around z-axis to go between global frame (base) and local frame (arm)
                z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.base_pose[2])
                z_rot_inv = z_rot.inv()
                ref_z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2])

                # Position
                pos_diff = pos - self.arm_xr_ref_pos  # WebXR
                pos_diff += ref_z_rot.apply(self.arm_ref_pos) - z_rot.apply(self.arm_ref_pos)  # Secondary base control: Compensate for base rotation
                pos_diff[:2] += self.arm_ref_base_pose[:2] - self.base_pose[:2]  # Secondary base control: Compensate for base translation
                self.arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)

                # Orientation
                self.arm_target_rot = (z_rot_inv * (rot * self.arm_xr_ref_rot_inv) * ref_z_rot) * self.arm_ref_rot

                # Gripper position
                self.gripper_target_pos = np.clip(self.gripper_ref_pos + data['gripper_delta'], 0.0, 1.0)

        # Teleop is disabled
        elif self.primary_device_id is None:
            # Update target pose in case base is pushed while teleop is disabled
            self.base_target_pose = self.base_pose

    def step(self, obs):
        """Generate action from current teleop targets.
        
        Updates robot state from observation, initializes targets on first call,
        and returns action dictionary with target poses for base, arm, and gripper.
        
        Args:
            obs: Dictionary containing current robot state observations
            
        Returns:
            Dictionary with 'base_pose', 'arm_pos', 'arm_quat', 'gripper_pos' or None if disabled
        """
        # Update robot state
        self.base_pose = obs['base_pose']

        # Initialize targets
        if not self.targets_initialized:
            self.base_target_pose = obs['base_pose']
            self.arm_target_pos = obs['arm_pos']
            self.arm_target_rot = R.from_quat(obs['arm_quat'])
            self.gripper_target_pos = obs['gripper_pos']
            self.targets_initialized = True

        # Return no action if teleop is not enabled
        if self.primary_device_id is None:
            return None

        # Get most recent teleop command
        arm_quat = self.arm_target_rot.as_quat()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness (Note: Not strictly necessary since policy training uses 6D rotation representation)
            np.negative(arm_quat, out=arm_quat)
        action = {
            'base_pose': self.base_target_pose.copy(),
            'arm_pos': self.arm_target_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_target_pos.copy(),
        }

        return action

# Teleop using WebXR phone web app
class TeleopPolicy(Policy):
    """Policy for teleoperating robot using WebXR phone interface."""
    
    def __init__(self, enable_web_server=True):
        """Initialize teleop policy with optional web server.
        
        Args:
            enable_web_server: If True, start Flask server for WebXR interface
        """
        self.web_server_queue = Queue()
        self.teleop_controller = None
        self.teleop_state = None  # States: episode_started -> episode_ended -> reset_env
        self.episode_ended = False
        self.enable_web_server = enable_web_server

        if self.enable_web_server:
            # Web server for serving the WebXR phone web app
            server = WebServer(self.web_server_queue)
            threading.Thread(target=server.run, daemon=True).start()
            print("Web server started for teleop interface")
        else:
            print("Web server disabled (simulation mode)")

        # Listener thread to process messages from WebXR client
        threading.Thread(target=self.listener_loop, daemon=True).start()

    def reset(self):
        """Reset policy for new episode and wait for user to start."""
        self.teleop_controller = TeleopController()
        self.episode_ended = False

        if self.enable_web_server:
            # Wait for user to signal that episode has started
            self.teleop_state = None
            while self.teleop_state != 'episode_started':
                time.sleep(0.01)
        else:
            # In sim mode without web server, start episode immediately
            self.teleop_state = 'episode_started'

    def step(self, obs):
        """Execute one policy step, handling episode state transitions.
        
        Args:
            obs: Dictionary containing robot state observations
            
        Returns:
            Action dictionary, 'end_episode', 'reset_env', or None
        """
        # Signal that user has ended episode
        if not self.episode_ended and self.teleop_state == 'episode_ended':
            self.episode_ended = True
            return 'end_episode'

        # Signal that user is ready for env reset (after ending the episode)
        if self.teleop_state == 'reset_env':
            return 'reset_env'

        return self._step(obs)

    def _step(self, obs):
        """Internal step method that delegates to teleop controller.
        
        Args:
            obs: Dictionary containing robot state observations
            
        Returns:
            Action dictionary from teleop controller
        """
        return self.teleop_controller.step(obs)

    def listener_loop(self):
        """Background thread loop that processes messages from WebXR client queue."""
        while True:
            if not self.web_server_queue.empty():
                data = self.web_server_queue.get()

                # Update state
                if 'state_update' in data:
                    self.teleop_state = data['state_update']

                # Process message if not stale
                elif 1000 * time.time() - data['timestamp'] < 250:  # 250 ms
                    self._process_message(data)

            time.sleep(0.001)

    def _process_message(self, data):
        """Process WebXR message through teleop controller.
        
        Args:
            data: Dictionary containing WebXR teleop data
        """
        self.teleop_controller.process_message(data)

# Execute policy running on remote server
class RemotePolicy(TeleopPolicy):
    """Policy that executes actions from a remote policy server via ZMQ."""
    
    def __init__(self, enable_web_server=True):
        """Initialize remote policy with connection to policy server.
        
        Args:
            enable_web_server: If True, enable WebXR interface for enabling policy
        """
        super().__init__(enable_web_server=enable_web_server)

        # Use phone as enabling device during policy rollout
        self.enabled = False

        # Connection to policy server
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')
        print(f'Connected to policy server at {POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')

    def reset(self):
        """Reset policy and remote server, wait for user enable signal."""
        # Wait for user to signal that episode has started (or skip if web server disabled)
        super().reset()

        # Check connection to policy server and reset policy
        default_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # Temporarily set 1000 ms timeout
        self.socket.send_pyobj({'reset': True})
        try:
            self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        except zmq.error.Again as e:
            raise Exception('Could not communicate with policy server') from e
        self.socket.setsockopt(zmq.RCVTIMEO, default_timeout)  # Put default timeout back

        # Disable policy execution until user presses on screen (or enable immediately in sim mode)
        self.enabled = not self.enable_web_server

    def _step(self, obs):
        """Execute one step by sending observation to remote policy server.
        
        Encodes images as JPEG and sends to server via ZMQ. Falls back to teleop
        if episode has ended.
        
        Args:
            obs: Dictionary containing robot state observations (may include images)
            
        Returns:
            Action dictionary from policy server, teleop action, or None
        """
        # Return teleop command if episode has ended
        if self.episode_ended:
            return self.teleop_controller.step(obs)

        # Return no action if robot is not enabled
        if not self.enabled:
            return None

        # Encode images
        encoded_obs = {}
        for k, v in obs.items():
            if v.ndim == 3:
                # Resize image to resolution expected by policy server
                v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))

                # Encode image as JPEG
                _, v = cv.imencode('.jpg', v)  # Note: Interprets RGB as BGR
                encoded_obs[k] = v
            else:
                encoded_obs[k] = v

        # Send obs to policy server
        req = {'obs': encoded_obs}
        self.socket.send_pyobj(req)

        # Get action from policy server
        rep = self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        action = rep['action']

        return action

    def _process_message(self, data):
        """Process WebXR message to enable/disable policy or handle teleop.
        
        Args:
            data: Dictionary containing WebXR teleop data
        """
        if self.episode_ended:
            # Run teleop controller if episode has ended
            self.teleop_controller.process_message(data)
        else:
            # Enable policy execution if user is pressing on screen
            self.enabled = 'teleop_mode' in data

# Motion Planner generated plan. 
class MotionPlannerPolicy(Policy):
    """Policy that executes motion plans for pick-and-place tasks using waypoint following."""
    
    def __init__(self):
        """Initialize motion planner policy with state machine and control parameters."""
        # Motion planning state - following controller.py pattern
        self.state = 'idle'  # States: idle, moving, manipulating, grasping
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.grasp_step = 0  # 0: position arm with open gripper, 1: close gripper
        
        # Arm mounting offset from base center (from tidybot.xml line 110)
        self.ARM_MOUNT_OFFSET = np.array([0.1199, 0.0, 0.3948])  # x, y, z offset in base frame
        
        # Base following parameters (from BaseController)
        self.LOOKAHEAD_DISTANCE = 0.3  # 30 cm
        self.lookahead_position = None
        self.position_tolerance = 0.005  # 0.5 cm (reduced from 1.5 cm)
        self.heading_tolerance = math.radians(2.1)  # 2.1 degrees
        
        # Object and target locations (using ground truth from MuJoCo)
        self.object_location = None
        self.target_location = None
        self.object_to_push_ids = [] #added for push primitive
        self.center_object_id = None #added for push primitive
        
        # Enable policy execution immediately (no web interface required)
        self.enabled = True
        self.episode_ended = False
        
        print(f'Motion planner policy initialized - ready to start automatically')

    def reset(self):
        """Reset motion planner state for a new episode."""
        # Reset motion planning state
        self.state = 'idle'
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.lookahead_position = None
        self.episode_ended = False
        self.grasp_step = 0  # Reset grasping step
        self.object_to_push_ids = [] #reset for push queue
        self.center_object_id = None #reset for center object
        
        # Clean up any grasp tracking variables
        if hasattr(self, 'grasp_start_time'):
            delattr(self, 'grasp_start_time')
        if hasattr(self, 'initial_gripper_pos'):
            delattr(self, 'initial_gripper_pos')
        
        # Enable policy execution immediately
        self.enabled = True
        
        print("Motion planner reset - starting episode automatically")

    def step(self, obs):
        """Execute one policy step, checking episode state and enabled status.
        
        Args:
            obs: Dictionary containing robot state observations
            
        Returns:
            Action dictionary, 'end_episode', or None
        """
        # Return no action if episode has ended
        if self.episode_ended:
            return 'end_episode'

        # Return no action if robot is not enabled
        if not self.enabled:
            return None

        return self._step(obs)

    def _step(self, obs):
        """Main state machine for motion planning execution.
        
        States: idle -> moving -> manipulating -> (back to idle or end)
        Handles object detection, base movement, and arm manipulation.
        
        Args:
            obs: Dictionary containing robot state observations
            
        Returns:
            Action dictionary with target poses
        """
        # Extract current state
        base_pose = obs['base_pose']
        arm_pos = obs['arm_pos'] 
        arm_quat = obs['arm_quat']
        gripper_pos = obs['gripper_pos']

        # Debug: Print current base pose
        print(f"Current base pose: [{base_pose[0]:.3f}, {base_pose[1]:.3f}, {base_pose[2]:.3f}]")

        # State machine following controller.py pattern
        if self.state == 'idle':
            # Detect objects and plan new command
            if not self.object_to_push_ids and self.center_object_id is None:
                #get all threee objects
                all_cubes = []
                for i in range(1,4):
                    cube_key = f'cube{i}_pos'
                    if cube_key in obs:
                        cube_pos = obs[cube_key].copy()
                        all_cubes.append((cube_pos, i))
                        print(f"Cube {i} detected at: {cube_pos}")
                if len(all_cubes) == 3:
                    #sort by x coordinate and select the middle one as the center object
                    all_cubes.sort(key=lambda x: x[0][0])
                    center_pos, center_id = all_cubes[1] #middle object
                    self.center_object_id = center_id
                    self.target_location = center_pos.copy() #use existing target location for center
                    #setup push queue with the other two objects
                    self.object_to_push_ids = [all_cubes[0][1], all_cubes[2][1]]
                    print("Initialized push sequence:")
                    print(f"Center object ID: {self.center_object_id} at {center_pos}")
                    print(f"Object to push IDs: {self.object_to_push_ids[0]} and {self.object_to_push_ids[1]}")
                else:
                    print(f"Only {len(all_cubes)} objects detected, expected 3")
                    return self.search_for_objects(obs)

            if self.object_to_push_ids:
                # Create push command for next object 
                next_object_id = self.object_to_push_ids[0]
                cube_key = f'cube{next_object_id}_pos'

                if cube_key not in obs:
                    print("Warning: cube{next_object_id}_pos not in observation")
                    self.object_to_push_ids.pop(0)
                    return None

                object_pos = obs[cube_key].copy()
                center_key = f'cube{self.center_object_id}_pos'
                if center_key in obs:
                    self.target_location = obs[center_key].copy()

                self.object_location = object_pos.copy()
                
                # 计算推进方向：从 object 到 target（center）
                push_direction_2d = self.target_location[:2] - self.object_location[:2]
                push_distance = np.linalg.norm(push_direction_2d)
                
                if push_distance < 0.01:
                    print("Object already at target location")
                    self.object_to_push_ids.pop(0)
                    return None
                
                push_direction_2d_normalized = push_direction_2d / push_distance
                
                # 计算推进角度（用于后续 gripper 方向）
                push_angle = math.atan2(push_direction_2d_normalized[1], push_direction_2d_normalized[0])
                
                # 关键：base 应该在推进方向的反向延长线上
                # 推进方向：object -> target（center）
                # 反向：target -> object（从 center 指向 object）
                # base 应该在 object 的后方，在反向延长线上
                # 增加额外距离，让机械臂能完全展开
                end_effector_offset = self.get_end_effector_offset('push')
                EXTRA_OFFSET = 0.0  # 额外15cm，让机械臂能完全展开
                total_offset = end_effector_offset + EXTRA_OFFSET
                base_target_2d = self.object_location[:2] - push_direction_2d_normalized * total_offset
                
                push_command = {
                    'primitive_name': 'push',
                    'waypoints': [base_pose[:2].tolist(), base_target_2d.tolist()],  # base 移动到反向延长线上（更远）
                    'object_3d_pos': self.object_location.copy(),
                    'target_3d_pos': self.target_location.copy(),
                    'object_id': next_object_id,
                    'push_angle': push_angle,  # 存储推进角度
                    'push_direction': push_direction_2d_normalized  # 存储推进方向
                }
                
                print(f"Pushing Object {next_object_id} at: {self.object_location}")
                print(f"Target push location (center): {self.target_location}")
                print(f"Push direction: {push_direction_2d_normalized}, angle: {math.degrees(push_angle):.1f} deg")
                print(f"Base target position (on reverse extension line, {total_offset:.3f}m from object): {base_target_2d}")
                
                # Build base command and start moving
                base_command = self.build_base_command(push_command)
                if base_command:
                    self.current_command = push_command
                    self.base_waypoints = base_command['waypoints']
                    self.target_ee_pos = base_command['target_ee_pos']
                    self.current_waypoint_idx = 1
                    self.lookahead_position = None
                    self.state = 'moving'
                    print(f"Starting base movement to object at {self.object_location}")
                    print(f"Base waypoints: {self.base_waypoints}")
                    print(f"Target EE position: {self.target_ee_pos}")
                else:
                    print("Failed to build base command")
                    self.object_to_push_ids.pop(0)
            else:
                #all objects pushed, end episode
                print("All objects pushed, ending episode")
                self.episode_ended = True
                return None
                
        elif self.state == 'moving':
            # Execute base movement following waypoints (like BaseController)
            action = self.execute_base_movement(obs)
            if action is None:  # Base movement complete
                print("Base movement complete!")
                
                # Check if we're close enough for arm manipulation
                if self.target_ee_pos is not None:
                    distance_to_target = self.distance(base_pose[:2], self.target_ee_pos)
                    end_effector_offset = self.get_end_effector_offset(self.current_command['primitive_name'])
                    diff = abs(end_effector_offset - distance_to_target)
                    print(f"Distance to target EE: {distance_to_target:.3f}, EE offset: {end_effector_offset:.3f}, diff: {diff:.3f}")
                    if self.current_command['primitive_name'] == 'place':
                        if diff < 0.05:  # 0.2 cm tolerance (reduced from 10 cm)
                            self.state = 'manipulating'
                            print("Base reached target, starting arm manipulation")
                    elif self.current_command['primitive_name'] == 'pick':
                        if diff < 0.005:  # 0.2 cm tolerance (reduced from 10 cm)
                            self.state = 'manipulating'
                            print("Base reached target, starting arm manipulation")
                    elif self.current_command['primitive_name'] == 'push':
                        if diff < 0.05:  # 5cm tolerance for push
                            self.state = 'manipulating'
                            self.grasp_step = 0  # Reset grasp_step when starting push manipulation
                            print("Base reached target, starting push manipulation (grasp_step reset to 0)")
                            #breakpoint()
                        else:
                            print(f'Too far from target end effector position ({(100 * diff):.1f} cm)')
                            self.state = 'idle'
                    else:
                        print(f"Unknown primitive name: {self.current_command['primitive_name']}")
                        self.state = 'idle'
                else:
                    self.state = 'idle'
            else:
                # Print target base pose from action
                if action and 'base_pose' in action:
                    target_pose = action['base_pose']
                    print(f"Target base pose: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}]")
            return action
            
        elif self.state == 'manipulating':
            # Execute arm manipulation
            # Important: All target positions are calculated relative to the arm mount point,
            # but actions use arm_pos which is relative to base center (controller expects base-relative coords)
            if self.current_command['primitive_name'] == 'pick':
                # Position arm above object and close gripper to grasp
                # Use actual detected object 3D position
                if 'object_3d_pos' in self.current_command:
                    object_3d_pos = self.current_command['object_3d_pos']
                    
                    # Calculate arm mount position in global frame (accounting for base rotation)
                    base_angle = base_pose[2]
                    cos_angle = math.cos(base_angle)
                    sin_angle = math.sin(base_angle)
                    arm_mount_global = np.array([
                        base_pose[0] + cos_angle * self.ARM_MOUNT_OFFSET[0] - sin_angle * self.ARM_MOUNT_OFFSET[1],
                        base_pose[1] + sin_angle * self.ARM_MOUNT_OFFSET[0] + cos_angle * self.ARM_MOUNT_OFFSET[1],
                        base_pose[2]  # Not used for Z calculation
                    ])
                    
                    # Calculate global position difference from arm mount point
                    global_diff = np.array([
                        object_3d_pos[0] - arm_mount_global[0],
                        object_3d_pos[1] - arm_mount_global[1], 
                        object_3d_pos[2] + 0.25 - 0.48 - 0.053  # Object height + higher offset - arm mount height
                    ])
                    
                    # Transform to base's local coordinate frame (account for base rotation)
                    cos_angle_inv = math.cos(-base_angle)  # Negative for inverse rotation
                    sin_angle_inv = math.sin(-base_angle)
                    
                    object_relative_pos = np.array([
                        cos_angle_inv * global_diff[0] - sin_angle_inv * global_diff[1],
                        sin_angle_inv * global_diff[0] + cos_angle_inv * global_diff[1],
                        global_diff[2]  # Z component unchanged
                    ])
                    print(f"Primary path: arm_mount_global = {arm_mount_global}, global_diff = {global_diff}")
                else:
                    # Fallback to target_ee_pos if object_3d_pos not available
                    # Calculate arm mount position in global frame
                    base_angle = base_pose[2]
                    cos_angle = math.cos(base_angle)
                    sin_angle = math.sin(base_angle)
                    arm_mount_global = np.array([
                        base_pose[0] + cos_angle * self.ARM_MOUNT_OFFSET[0] - sin_angle * self.ARM_MOUNT_OFFSET[1],
                        base_pose[1] + sin_angle * self.ARM_MOUNT_OFFSET[0] + cos_angle * self.ARM_MOUNT_OFFSET[1],
                        base_pose[2]
                    ])
                    
                    global_diff = np.array([
                        self.target_ee_pos[0] - arm_mount_global[0],
                        self.target_ee_pos[1] - arm_mount_global[1], 
                        -0.33 - 0.053  # Slightly higher position for better top grasp approach
                    ])
                    
                    # Transform to base's local coordinate frame
                    cos_angle_inv = math.cos(-base_angle)
                    sin_angle_inv = math.sin(-base_angle)
                    
                    object_relative_pos = np.array([
                        cos_angle_inv * global_diff[0] - sin_angle_inv * global_diff[1],
                        sin_angle_inv * global_diff[0] + cos_angle_inv * global_diff[1],
                        global_diff[2]
                    ])
                    print(f"Fallback path: arm_mount_global = {arm_mount_global}, global_diff = {global_diff}")
                
                # Transform arm_pos to be relative to arm mount for accurate comparison
                # (arm_pos from obs is relative to base center, object_relative_pos is relative to arm mount)
                arm_pos_from_mount = arm_pos - np.array([self.ARM_MOUNT_OFFSET[0], self.ARM_MOUNT_OFFSET[1], 0])
                
                print(f"Base angle: {base_pose[2]:.3f} rad ({math.degrees(base_pose[2]):.1f} deg)")
                print(f"Object global pos: {self.current_command.get('object_3d_pos', 'N/A')}")
                print(f"Transformed object_relative_pos (from arm mount) = {object_relative_pos}")
                print(f"Current arm pos (from base center): {arm_pos}")
                print(f"Current arm pos (from arm mount): {arm_pos_from_mount}")
                print(f"Position error: {np.linalg.norm(arm_pos_from_mount - object_relative_pos):.4f}m")
                print(f"Grasp step: {self.grasp_step}")
                
                if self.grasp_step == 0:
                    # Step 1: 确定末端抓夹方向（向前为正向）与推进方向形成锐角
                    # 为了让机械臂向外展开，末端位置应该更靠前（在推进方向上）
                    push_angle_global = push_angle + base_angle
                    
                    # 创建旋转：gripper 向前倾斜，与推进方向形成锐角
                    z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * push_angle_global)
                    ACUTE_ANGLE = math.radians(30)  # 30度锐角
                    y_rot_horizontal = R.from_rotvec(np.array([0.0, 1.0, 0.0]) * (-math.pi / 2))
                    x_rot_tilt = R.from_rotvec(np.array([1.0, 0.0, 0.0]) * (-ACUTE_ANGLE))
                    gripper_quat = (z_rot * y_rot_horizontal * x_rot_tilt).as_quat()
                    
                    # 关键修改：让末端位置更靠前，引导机械臂向外展开
                    # 计算物体相对于base的位置
                    object_relative_to_base = object_3d_pos[:2] - base_pose[:2]
                    object_dist = np.linalg.norm(object_relative_to_base)
                    
                    # 如果物体较远，让末端位置在物体前方（推进方向上），引导机械臂向外展开
                    # 如果物体较近，让末端位置在物体后方，便于包含
                    if object_dist > 0.35:  # 物体距离base较远（>35cm）
                        approach_offset = 0.03  # 3cm在物体前方，引导向外展开
                    else:  # 物体较近
                        approach_offset = -0.02  # 2cm在物体后方，便于包含
                    
                    approach_pos_2d = object_3d_pos[:2] + push_direction_2d_normalized * approach_offset
                    
                    global_diff = np.array([
                        approach_pos_2d[0] - arm_mount_global[0],
                        approach_pos_2d[1] - arm_mount_global[1],
                        object_3d_pos[2] + 0.10 - self.ARM_MOUNT_OFFSET[2] - 0.053
                    ])
                    
                    cos_angle_inv = math.cos(-base_angle)
                    sin_angle_inv = math.sin(-base_angle)
                    
                    object_relative_pos = np.array([
                        cos_angle_inv * global_diff[0] - sin_angle_inv * global_diff[1],
                        sin_angle_inv * global_diff[0] + cos_angle_inv * global_diff[1],
                        global_diff[2]
                    ])
                    
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': object_relative_pos,
                        'arm_quat': gripper_quat,
                        'gripper_pos': np.array([0.0])
                    }
                    
                    print(f"Step 1: Positioning arm above object, gripper forward-tilted")
                    print(f"Push angle: {math.degrees(push_angle):.1f} deg, Object dist: {object_dist:.3f}m")
                    print(f"Approach offset: {approach_offset:.3f}m (positive=forward, negative=backward)")
                    
                    if np.allclose(arm_pos_from_mount, object_relative_pos, atol=0.03):
                        self.grasp_step = 1
                        print("Arm positioned above object, ready to contain object")
                
                elif self.grasp_step == 1:
                    # Step 2: Lower gripper closer to object for precise grasping
                    lower_pos = object_relative_pos.copy()
                    lower_pos[2] -= 0.08  # Lower by 8cm for closer approach (more conservative)
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': lower_pos,  # Lower the arm closer to object
                        'arm_quat': np.array([1.0, 0.0, 0.0, 0.0]),  # Inverted: gripper fingers pointing down toward ground
                        'gripper_pos': np.array([0.0])  # Keep gripper open while lowering
                    }
                    print(f"Step 2: Lowering gripper for precise approach... target: {lower_pos[2]:.3f}, current: {arm_pos_from_mount[2]:.3f}")
                    if np.allclose(arm_pos_from_mount, lower_pos, atol=0.02):  # Very tight tolerance: 2cm
                        self.grasp_step = 2
                        print("Gripper lowered to grasping position, closing gripper")
                
                elif self.grasp_step == 2:
                    # Step 3: Close gripper to grasp
                    lower_pos = object_relative_pos.copy()
                    lower_pos[2] -= 0.08  # Maintain lowered position
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': lower_pos,  # Maintain lowered position
                        'arm_quat': np.array([1.0, 0.0, 0.0, 0.0]),  # Inverted: gripper fingers pointing down toward ground
                        'gripper_pos': np.array([1.0])  # Close gripper
                    }
                    print(f"Step 3: Closing gripper... current position: {gripper_pos[0]:.3f}")
                    
                    # Initialize grasp attempt tracking
                    if not hasattr(self, 'grasp_start_time'):
                        self.grasp_start_time = time.time()
                        self.initial_gripper_pos = gripper_pos[0]
                        print(f"Started grasp attempt, initial gripper pos: {self.initial_gripper_pos:.3f}")
                    
                    # Check for successful grasp (multiple criteria)
                    gripper_closed_enough = gripper_pos[0] > 0.55  # Lower threshold (was 0.8)
                    gripper_progress = (gripper_pos[0] - self.initial_gripper_pos) > 0.3  # Made significant progress
                    grasp_timeout = (time.time() - self.grasp_start_time) > 3.0  # 3 second timeout
                    
                    if gripper_closed_enough or gripper_progress or grasp_timeout:
                        if gripper_closed_enough or gripper_progress:
                            print(f"Grasp successful! Gripper pos: {gripper_pos[0]:.3f}, progress: {gripper_pos[0] - self.initial_gripper_pos:.3f}")
                        else:
                            print(f"Grasp timeout reached, proceeding with current grip: {gripper_pos[0]:.3f}")
                        
                        self.grasp_step = 3
                        # Clean up tracking variables
                        delattr(self, 'grasp_start_time')
                        delattr(self, 'initial_gripper_pos')
                        print("Moving to lift phase!")
                
                elif self.grasp_step == 3:
                    # Step 4: Lift object by 20cm from the grasping position
                    lower_pos = object_relative_pos.copy()
                    lower_pos[2] -= 0.08  # Start from lowered grasping position
                    lifted_pos = lower_pos.copy()
                    lifted_pos[2] += 0.28  # Lift by 28cm from grasping position (net +20cm from original)
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': lifted_pos,  # Lift the object
                        'arm_quat': np.array([1.0, 0.0, 0.0, 0.0]),  # Inverted: gripper fingers pointing down toward ground
                        'gripper_pos': np.array([1.0])  # Keep gripper closed
                    }
                    print(f"Step 4: Lifting object... target height: {lifted_pos[2]:.3f}, current: {arm_pos_from_mount[2]:.3f}")
                    if np.allclose(arm_pos_from_mount, lifted_pos, atol=0.05):  # 5cm tolerance
                        print("Object lifted successfully! Now moving to placement location.")
                        # Create place command to move to placement location (already set dynamically)
                        place_command = {
                            'primitive_name': 'place',
                            'waypoints': [base_pose[:2].tolist(), self.target_location[:2].tolist()],
                            'target_3d_pos': self.target_location.copy()  # Store full 3D target position
                        }
                        
                        # Build base command for placement
                        base_command = self.build_base_command(place_command)
                        if base_command:
                            self.current_command = place_command
                            self.base_waypoints = base_command['waypoints']
                            self.target_ee_pos = base_command['target_ee_pos']
                            self.current_waypoint_idx = 1
                            self.lookahead_position = None
                            self.state = 'moving'
                            self.grasp_step = 0  # Reset for next manipulation
                            print(f"Starting base movement to placement location at {self.target_location}")
                        else:
                            print("Failed to build place command")
                            self.episode_ended = True
                            self.state = 'idle'
                
                return action
            elif self.current_command['primitive_name'] == 'place':
                # Position arm above placement location and open gripper
                # Use actual target 3D position
                if 'target_3d_pos' in self.current_command:
                    target_3d_pos = self.current_command['target_3d_pos']
                    
                    # Calculate arm mount position in global frame (accounting for base rotation)
                    base_angle = base_pose[2]
                    cos_angle = math.cos(base_angle)
                    sin_angle = math.sin(base_angle)
                    arm_mount_global = np.array([
                        base_pose[0] + cos_angle * self.ARM_MOUNT_OFFSET[0] - sin_angle * self.ARM_MOUNT_OFFSET[1],
                        base_pose[1] + sin_angle * self.ARM_MOUNT_OFFSET[0] + cos_angle * self.ARM_MOUNT_OFFSET[1],
                        base_pose[2]  # Not used for Z calculation
                    ])
                    
                    # Calculate global position difference from arm mount point
                    global_diff = np.array([
                        target_3d_pos[0] - arm_mount_global[0],
                        target_3d_pos[1] - arm_mount_global[1], 
                        target_3d_pos[2] + 0.10 - self.ARM_MOUNT_OFFSET[2] - 0.053  # Target height + offset - arm mount height
                    ])
                    
                    # Transform to base's local coordinate frame (account for base rotation)
                    cos_angle_inv = math.cos(-base_angle)  # Negative for inverse rotation
                    sin_angle_inv = math.sin(-base_angle)
                    
                    target_relative_pos = np.array([
                        cos_angle_inv * global_diff[0] - sin_angle_inv * global_diff[1],
                        sin_angle_inv * global_diff[0] + cos_angle_inv * global_diff[1],
                        global_diff[2]  # Z component unchanged
                    ])
                else:
                    # Fallback to target_ee_pos if target_3d_pos not available
                    # Calculate arm mount position in global frame
                    base_angle = base_pose[2]
                    cos_angle = math.cos(base_angle)
                    sin_angle = math.sin(base_angle)
                    arm_mount_global = np.array([
                        base_pose[0] + cos_angle * self.ARM_MOUNT_OFFSET[0] - sin_angle * self.ARM_MOUNT_OFFSET[1],
                        base_pose[1] + sin_angle * self.ARM_MOUNT_OFFSET[0] + cos_angle * self.ARM_MOUNT_OFFSET[1],
                        base_pose[2]
                    ])
                    
                    global_diff = np.array([
                        self.target_ee_pos[0] - arm_mount_global[0],
                        self.target_ee_pos[1] - arm_mount_global[1], 
                        -0.30  # Position slightly above target for placement
                    ])
                    
                    # Transform to base's local coordinate frame
                    cos_angle_inv = math.cos(-base_angle)
                    sin_angle_inv = math.sin(-base_angle)
                    
                    target_relative_pos = np.array([
                        cos_angle_inv * global_diff[0] - sin_angle_inv * global_diff[1],
                        sin_angle_inv * global_diff[0] + cos_angle_inv * global_diff[1],
                        global_diff[2]
                    ])
                
                # Transform arm_pos to be relative to arm mount for accurate comparison
                arm_pos_from_mount = arm_pos - np.array([self.ARM_MOUNT_OFFSET[0], self.ARM_MOUNT_OFFSET[1], 0])
                
                print(f"Placing: target_relative_pos (from arm mount) = {target_relative_pos}")
                print(f"Current arm pos (from base center): {arm_pos}")
                print(f"Current arm pos (from arm mount): {arm_pos_from_mount}")
                print(f"Target EE pos: {self.target_ee_pos}, Base pose: {base_pose}")
                print(f"Position error: {np.linalg.norm(arm_pos_from_mount - target_relative_pos):.4f}m")
                print(f"Grasp step: {self.grasp_step}")
                
                if self.grasp_step == 0:
                    # Step 1: Position arm above placement location with closed gripper
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': target_relative_pos,  # Position arm above placement location
                        'arm_quat': np.array([1.0, 0.0, 0.0, 0.0]),  # Inverted: gripper fingers pointing down toward ground
                        'gripper_pos': np.array([1.0])  # Keep gripper closed while positioning
                    }
                    print(f"Step 1: Positioning arm above placement location with closed gripper")
                    # Move to next step after arm is positioned
                    if np.allclose(arm_pos_from_mount, target_relative_pos, atol=0.05):  # 5cm tolerance
                        self.grasp_step = 1
                        print("Arm positioned above placement location, opening gripper")
                
                elif self.grasp_step == 1:
                    # Step 2: Open gripper to place object
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': target_relative_pos,  # Maintain arm position
                        'arm_quat': np.array([1.0, 0.0, 0.0, 0.0]),  # Inverted: gripper fingers pointing down toward ground
                        'gripper_pos': np.array([0.0])  # Open gripper
                    }
                    print(f"Step 2: Opening gripper... current position: {gripper_pos[0]:.3f}")
                    if gripper_pos[0] < 0.2:  # Gripper opened successfully
                        print("Object placed successfully! Task complete.")
                        self.episode_ended = True  # End the episode
                        self.state = 'idle'
                
                return action
            elif self.current_command['primitive_name'] == 'push':
                # Push operation: push object to target location
                # IMPORTANT: Get latest object position from obs, not from stored command
                # This ensures we use the current position after base movement
                object_id = self.current_command.get('object_id')
                cube_key = f'cube{object_id}_pos'
                center_key = f'cube{self.center_object_id}_pos'
                
                if cube_key not in obs:
                    print(f"Error: {cube_key} not in observation")
                    self.state = 'idle'
                    return None
                
                # Get latest positions from observation
                object_3d_pos = obs[cube_key].copy()
                if center_key in obs:
                    target_3d_pos = obs[center_key].copy()
                else:
                    # Fallback to stored target if not in obs
                    target_3d_pos = self.current_command.get('target_3d_pos')
                    if target_3d_pos is None:
                        print("Error: Target position not available")
                        self.state = 'idle'
                        return None
                
                print(f"Push operation: Using latest object position from obs: {object_3d_pos}")
                print(f"Target position: {target_3d_pos}")
                
                # Initialize grasp_step if not already set (should be 0, but ensure it's reset)
                if not hasattr(self, 'grasp_step') or self.grasp_step is None:
                    self.grasp_step = 0
                    print("Initialized grasp_step to 0 for push operation")
                
                # Calculate push direction (from object to target)
                push_direction_2d = target_3d_pos[:2] - object_3d_pos[:2]
                push_distance = np.linalg.norm(push_direction_2d)
                
                if push_distance < 0.01:  # Already at target
                    print("Object already at target location")
                    self.state = 'idle'
                    return None
                
                push_direction_2d_normalized = push_direction_2d / push_distance
                
                # Calculate gripper orientation (aligned with push direction)
                # Gripper should point in push direction
                push_angle = math.atan2(push_direction_2d_normalized[1], push_direction_2d_normalized[0])
                
                # Calculate object position relative to arm mount point
                base_angle = base_pose[2]
                cos_angle = math.cos(base_angle)
                sin_angle = math.sin(base_angle)
                arm_mount_global = np.array([
                    base_pose[0] + cos_angle * self.ARM_MOUNT_OFFSET[0] - sin_angle * self.ARM_MOUNT_OFFSET[1],
                    base_pose[1] + sin_angle * self.ARM_MOUNT_OFFSET[0] + cos_angle * self.ARM_MOUNT_OFFSET[1],
                    base_pose[2]  # Not used for Z calculation
                ])
                
                # Transform arm_pos to be relative to arm mount for accurate comparison
                arm_pos_from_mount = arm_pos - np.array([self.ARM_MOUNT_OFFSET[0], self.ARM_MOUNT_OFFSET[1], 0])
                if self.grasp_step == 0:
                    print('manual debug mode')
                    
                    # ========== 手动调试模式 ==========
                    # 使用 RPY (Roll, Pitch, Yaw) 角度来设置姿态，更直观！
                    # 修改下面的参数即可，支持度数或弧度
                    
                    # ===== 位置设置（相对于 base center，单位：米）=====
                    MANUAL_X = 0.5      # 前后方向：正数=向前，负数=向后
                    MANUAL_Y = 0.0      # 左右方向：正数=向左，负数=向右
                    MANUAL_Z = 0.0      # 上下方向：正数=向上，负数=向下
                    MANUAL_ARM_POS = np.array([MANUAL_X, MANUAL_Y, MANUAL_Z])
                    
                    # ===== 姿态设置（RPY 角度）=====
                    # 方式1：使用度数（更直观，推荐）
                    USE_DEGREES = True  # True=使用度数，False=使用弧度
                    
                    if USE_DEGREES:
                        # 使用度数设置（更直观）
                        # 常用姿态参考：
                        # - 手指水平向前：Roll=0, Pitch=0, Yaw=0
                        # - 手指朝下：Roll=0, Pitch=-90, Yaw=0
                        # - 手指朝上：Roll=0, Pitch=90, Yaw=0
                        # - 手指水平向右：Roll=0, Pitch=0, Yaw=90
                        # - 手指水平向左：Roll=0, Pitch=0, Yaw=-90
                        # - 手指水平向后：Roll=0, Pitch=0, Yaw=180
                        MANUAL_ROLL_DEG = 180    # 绕 x 轴旋转（度）：0=不旋转，±180=翻转
                        MANUAL_PITCH_DEG = 0.0   # 绕 y 轴旋转（度）：0=水平，-90=朝下，90=朝上
                        MANUAL_YAW_DEG = 0.0     # 绕 z 轴旋转（度）：0=向前，90=向右，-90=向左，180=向后
                        
                        # 转换为弧度
                        manual_roll = math.radians(MANUAL_ROLL_DEG)
                        manual_pitch = math.radians(MANUAL_PITCH_DEG)
                        manual_yaw = math.radians(MANUAL_YAW_DEG)
                    else:
                        # 使用弧度设置
                        manual_roll = 0.0        # 绕 x 轴旋转（弧度）
                        manual_pitch = 0.0       # 绕 y 轴旋转（弧度）：0=水平，-π/2=朝下
                        manual_yaw = 0.0         # 绕 z 轴旋转（弧度）：0=向前，π/2=向右
                    
                    # 将 RPY 转换为四元数
                    MANUAL_ARM_QUAT = R.from_euler('xyz', [manual_roll, manual_pitch, manual_yaw], degrees=False).as_quat()
                    
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': MANUAL_ARM_POS,
                        'arm_quat': MANUAL_ARM_QUAT,
                        'gripper_pos': np.array([0.0])
                    }
                    
                    print(f"=== 手动调试模式 ===")
                    print(f"位置 (相对于 base center): X={MANUAL_X:.3f}m, Y={MANUAL_Y:.3f}m, Z={MANUAL_Z:.3f}m")
                    if USE_DEGREES:
                        print(f"姿态 (RPY 度数): Roll={MANUAL_ROLL_DEG:.1f}°, Pitch={MANUAL_PITCH_DEG:.1f}°, Yaw={MANUAL_YAW_DEG:.1f}°")
                    else:
                        print(f"姿态 (RPY 弧度): Roll={manual_roll:.3f}, Pitch={manual_pitch:.3f}, Yaw={manual_yaw:.3f}")
                    print(f"四元数: {MANUAL_ARM_QUAT}")
                    print(f"\n当前机械臂状态:")
                    print(f"  位置: {arm_pos}")
                    print(f"  姿态: {arm_quat}")
                    print(f"  姿态(欧拉角): {R.from_quat(arm_quat).as_euler('xyz', degrees=True)}")
                    
                    return action
                    
                    # ========== 自动计算模式（正常使用） ==========
                    
                    # Step 1: 定位机械臂到物体上方，gripper 水平指向推进方向
                    push_angle_global = push_angle + base_angle
                    
                    # 创建水平姿态：gripper 水平指向推进方向
                    # 方法：先让 gripper 水平（从朝下旋转90度到水平），然后绕 z 轴旋转到推进方向
                    # 1. 从手指朝下 [1,0,0,0] 旋转到水平：绕 x 轴旋转 -90度（或 +90度，取决于坐标系）
                    # 2. 然后绕 z 轴旋转到推进方向
                    
                    # 先创建水平姿态（手指向前，不朝下也不朝上）
                    # 从手指朝下到水平：绕 x 轴旋转 -90度（或 y 轴旋转，取决于gripper坐标系定义）
                    # 使用 y 轴旋转：从朝下到水平向前
                    y_rot_to_horizontal = R.from_rotvec(np.array([0.0, 1.0, 0.0]) * (-math.pi / 2))  # 绕 y 轴旋转 -90度，从朝下到水平
                    # 然后绕 z 轴旋转到推进方向
                    z_rot_to_direction = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * push_angle_global)
                    # 组合旋转：先水平，再转向
                    gripper_quat = (z_rot_to_direction * y_rot_to_horizontal).as_quat()
                    
                    # 定位到物体上方（稍微靠后，以便后续包含物体）
                    # 位置应该在物体后方（推进方向的反向），这样 gripper 可以"抓住"物体
                    approach_offset = -0.05  # 5cm 在物体后方
                    approach_pos_2d = object_3d_pos[:2] + push_direction_2d_normalized * approach_offset
                    
                    global_diff = np.array([
                        approach_pos_2d[0] - arm_mount_global[0],
                        approach_pos_2d[1] - arm_mount_global[1],
                        object_3d_pos[2] + 0.10 - self.ARM_MOUNT_OFFSET[2] - 0.053  # 物体上方 10cm
                    ])
                    
                    cos_angle_inv = math.cos(-base_angle)
                    sin_angle_inv = math.sin(-base_angle)
                    
                    object_relative_pos = np.array([
                        cos_angle_inv * global_diff[0] - sin_angle_inv * global_diff[1],
                        sin_angle_inv * global_diff[0] + cos_angle_inv * global_diff[1],
                        global_diff[2]
                    ])
                    
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': object_relative_pos,  # 定位到物体上方（后方）
                        'arm_quat': gripper_quat,  # Gripper 向前倾斜，与推进方向形成锐角
                        'gripper_pos': np.array([0.0])  # 保持 gripper 打开
                    }
                    
                    # 计算位置误差用于调试
                    position_error = np.linalg.norm(arm_pos_from_mount - object_relative_pos)
                    
                    print(f"Step 1: Positioning arm above object, gripper horizontal pointing in push direction")
                    print(f"Push angle: {math.degrees(push_angle):.1f} deg, Global angle: {math.degrees(push_angle_global):.1f} deg")
                    print(f"Push distance: {push_distance:.3f}m")
                    print(f"Current arm pos (from mount): {arm_pos_from_mount}")
                    print(f"Target arm pos (from mount): {object_relative_pos}")
                    print(f"Position error: {position_error:.4f}m (tolerance: 0.03m)")
                    print(f"\n=== 手动调试：复制以下值到手动调试模式 ===")
                    print(f"# 当前机械臂位置（相对于 base center）")
                    print(f"MANUAL_ARM_POS = np.array({arm_pos.tolist()})")
                    print(f"# 当前机械臂姿态")
                    print(f"MANUAL_ARM_QUAT = np.array({arm_quat.tolist()})")
                    print(f"# 目标机械臂位置（相对于 base center）")
                    target_arm_pos_base = object_relative_pos + np.array([self.ARM_MOUNT_OFFSET[0], self.ARM_MOUNT_OFFSET[1], 0])
                    print(f"MANUAL_ARM_POS = np.array({target_arm_pos_base.tolist()})")
                    print(f"# 目标机械臂姿态")
                    print(f"MANUAL_ARM_QUAT = np.array({gripper_quat.tolist()})")
                    print(f"# 或者使用欧拉角（roll, pitch, yaw）：")
                    euler_angles = R.from_quat(gripper_quat).as_euler('xyz', degrees=False)
                    print(f"# roll={euler_angles[0]:.3f}, pitch={euler_angles[1]:.3f}, yaw={euler_angles[2]:.3f}")
                    print(f"# 或者（度）：roll={math.degrees(euler_angles[0]):.1f}°, pitch={math.degrees(euler_angles[1]):.1f}°, yaw={math.degrees(euler_angles[2]):.1f}°")
                    print(f"==========================================\n")
                    print(f"=== 调试信息：你可以复制这些值来手动测试 ===")
                    print(f"Current arm_pos (from base center): {arm_pos}")
                    print(f"Current arm_quat: {arm_quat}")
                    print(f"Target arm_pos (from base center): {object_relative_pos + np.array([self.ARM_MOUNT_OFFSET[0], self.ARM_MOUNT_OFFSET[1], 0])}")
                    print(f"Target arm_quat: {gripper_quat}")
                    
                    if np.allclose(arm_pos_from_mount, object_relative_pos, atol=0.03):  # 3cm tolerance
                        self.grasp_step = 1
                        print("Arm positioned above object, ready to contain object")
                    else:
                        # 如果误差太大，可能是目标位置不合理，放宽容差或给出警告
                        if position_error > 0.20:  # 如果误差超过20cm，可能是计算错误
                            print(f"WARNING: Position error is very large ({position_error:.3f}m). Target may be unreachable!")
                            print(f"Consider checking if object_relative_pos is within arm workspace.")
                
                elif self.grasp_step == 1:
                    # Step 2: 确认物体在抓夹内
                    # 降低到物体高度，闭合 gripper 使物体在抓夹内
                    push_angle_global = push_angle + base_angle
                    
                    # 保持水平姿态，指向推进方向（与 Step 1 相同）
                    y_rot_to_horizontal = R.from_rotvec(np.array([0.0, 1.0, 0.0]) * (-math.pi / 2))
                    z_rot_to_direction = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * push_angle_global)
                    gripper_quat = (z_rot_to_direction * y_rot_to_horizontal).as_quat()
                    
                    # 降低到物体高度，位置在物体中心或稍微靠后
                    # 关键：z轴位置应该在物体中心稍微偏下，这样gripper可以包含物体
                    contain_offset = -0.02  # 2cm 在物体后方，确保物体在抓夹内
                    contain_pos_2d = object_3d_pos[:2] + push_direction_2d_normalized * contain_offset
                    
                    # 修正z轴位置：物体中心高度减去一些偏移，让gripper能够包含物体
                    # object_3d_pos[2] 是物体中心，我们需要稍微低一点
                    CUBE_HEIGHT = 0.04  # 假设cube高度是4cm（从代码看是0.02*2）
                    object_bottom = object_3d_pos[2] - CUBE_HEIGHT / 2  # 物体底部高度
                    # Gripper应该在物体中心稍微偏下，或者接近物体底部
                    target_gripper_height = object_3d_pos[2] - 0.01  # 物体中心下方1cm
                    
                    global_diff = np.array([
                        contain_pos_2d[0] - arm_mount_global[0],
                        contain_pos_2d[1] - arm_mount_global[1],
                        target_gripper_height - self.ARM_MOUNT_OFFSET[2] - 0.053  # 修正后的高度
                    ])
                    
                    cos_angle_inv = math.cos(-base_angle)
                    sin_angle_inv = math.sin(-base_angle)
                    
                    contain_pos = np.array([
                        cos_angle_inv * global_diff[0] - sin_angle_inv * global_diff[1],
                        sin_angle_inv * global_diff[0] + cos_angle_inv * global_diff[1],
                        global_diff[2]
                    ])
                    
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': contain_pos,  # 降低到物体高度（稍微偏下）
                        'arm_quat': gripper_quat,  # 保持向前倾斜
                        'gripper_pos': np.array([0.6])  # 闭合 gripper，使物体在抓夹内
                    }
                    
                    print(f"Step 2: Lowering to object height and closing gripper to contain object...")
                    print(f"Object center height: {object_3d_pos[2]:.3f}, target gripper height: {target_gripper_height:.3f}")
                    print(f"Target arm pos z: {contain_pos[2]:.3f}, current: {arm_pos_from_mount[2]:.3f}")
                    if np.allclose(arm_pos_from_mount, contain_pos, atol=0.02):  # 2cm tolerance
                        self.grasp_step = 2
                        print("Object contained in gripper, ready to push")
                
                elif self.grasp_step == 2:
                    # Step 3: 向前进方向推行
                    # 保持 gripper 方向和闭合状态，向前推进
                    push_angle_global = push_angle + base_angle
                    
                    # 保持水平姿态，指向推进方向（与 Step 1 相同）
                    y_rot_to_horizontal = R.from_rotvec(np.array([0.0, 1.0, 0.0]) * (-math.pi / 2))
                    z_rot_to_direction = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * push_angle_global)
                    gripper_quat = (z_rot_to_direction * y_rot_to_horizontal).as_quat()
                    
                    # 向前推进：移动到目标位置
                    # 保持相同的高度（物体中心稍微偏下）
                    push_ahead = 0.05  # 5cm 超过目标
                    push_target_2d = target_3d_pos[:2] + push_direction_2d_normalized * push_ahead
                    
                    # 使用与 Step 2 相同的高度
                    target_gripper_height = object_3d_pos[2] - 0.01  # 保持相同高度
                    
                    global_diff = np.array([
                        push_target_2d[0] - arm_mount_global[0],
                        push_target_2d[1] - arm_mount_global[1],
                        target_gripper_height - self.ARM_MOUNT_OFFSET[2] - 0.053  # 保持相同高度
                    ])
                    
                    cos_angle_inv = math.cos(-base_angle)
                    sin_angle_inv = math.sin(-base_angle)
                    
                    push_end_pos = np.array([
                        cos_angle_inv * global_diff[0] - sin_angle_inv * global_diff[1],
                        sin_angle_inv * global_diff[0] + cos_angle_inv * global_diff[1],
                        global_diff[2]
                    ])
                    
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': push_end_pos,  # 向前移动到目标位置
                        'arm_quat': gripper_quat,  # 保持向前倾斜
                        'gripper_pos': np.array([0.6])  # 保持闭合，物体在抓夹内
                    }
                    
                    print(f"Step 3: Pushing forward... target: {push_end_pos[:2]}, current: {arm_pos_from_mount[:2]}")
                    print(f"Maintaining height: {push_end_pos[2]:.3f}")
                    distance_to_target = np.linalg.norm(arm_pos_from_mount[:2] - push_end_pos[:2])
                    if distance_to_target < 0.05:  # 5cm tolerance
                        self.grasp_step = 3
                        print("Push completed! Object should be at target location")

        
        action = {
            'base_pose': base_pose.copy(),
            'arm_pos': arm_pos.copy(),
            'arm_quat': arm_quat.copy(),
            'gripper_pos': gripper_pos.copy(),
        }
        print(f"Default action - holding current pose")
        return action

    def execute_base_movement(self, obs):
        """Execute base movement following waypoints using lookahead control.
        
        Implements pure pursuit-like controller that follows waypoints with
        lookahead distance. Computes target heading to face end effector target.
        
        Args:
            obs: Dictionary containing current robot state
            
        Returns:
            Action dictionary with target base pose, or None if movement complete
        """
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
            t2 = self.intersect(d, f, self.LOOKAHEAD_DISTANCE)
            
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
            position_error = self.distance(base_pose[:2], target_position)
            print(f"Position error to final target: {position_error:.3f} (tolerance: {self.position_tolerance})")
            if position_error < self.position_tolerance:
                print("Reached final position within tolerance")
                return None  # Movement complete
        else:
            target_position = self.lookahead_position
            print(f"Using lookahead as target: {target_position}")
        
        # Compute target heading
        target_heading = base_pose[2]
        if self.target_ee_pos is not None:
            # For push operation, base should face push direction (forward, toward push direction)
            if hasattr(self, 'current_command') and self.current_command and self.current_command.get('primitive_name') == 'push':
                # Get push angle from command (base should face push direction)
                push_angle = self.current_command.get('push_angle')
                if push_angle is not None:
                    target_heading = push_angle  # Base faces push direction
                    print(f"Push operation: Setting base heading to push direction: {math.degrees(push_angle):.1f} deg")
                else:
                    # Fallback: calculate from target_ee_pos
                    dx = self.target_ee_pos[0] - base_pose[0]
                    dy = self.target_ee_pos[1] - base_pose[1]
                    target_heading = math.atan2(dy, dx)
            else:
                # For other operations, turn to face target end effector position
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
                        remaining_path_length += self.distance(curr_waypoint, next_waypoint)
                        curr_waypoint = next_waypoint
                    frac = math.sqrt(self.LOOKAHEAD_DISTANCE / max(remaining_path_length, self.LOOKAHEAD_DISTANCE))
                
                heading_diff = self.restrict_heading_range(desired_heading - base_pose[2])
                target_heading = base_pose[2] + frac * heading_diff
                print(f"Heading: current={base_pose[2]:.3f}, desired={desired_heading:.3f}, diff={heading_diff:.3f}, frac={frac:.3f}, target={target_heading:.3f}")
        
        arm_pos_from_mount = obs['arm_pos'] - np.array([self.ARM_MOUNT_OFFSET[0], self.ARM_MOUNT_OFFSET[1], 0])
        # Create action to move towards target
        action = {
            'base_pose': np.array([target_position[0], target_position[1], target_heading]),
            'arm_pos': arm_pos_from_mount.copy(),
            'arm_quat': obs['arm_quat'].copy(),
            'gripper_pos': obs['gripper_pos'].copy(),
        }

        # import pdb; pdb.set_trace()
        
        return action

    def dot(self, a, b):
        """Compute 2D dot product of two vectors.
        
        Args:
            a: 2D vector [x, y]
            b: 2D vector [x, y]
            
        Returns:
            Scalar dot product
        """
        return a[0] * b[0] + a[1] * b[1]

    def intersect(self, d, f, r, use_t1=False):
        """Find intersection point between line segment and circle.
        
        Used for lookahead point calculation in waypoint following.
        Based on circle-line segment collision detection algorithm.
        
        Args:
            d: Direction vector of line segment
            f: Vector from circle center to line start
            r: Circle radius (lookahead distance)
            use_t1: If True, return first intersection; else return second
            
        Returns:
            Parameter t in [0,1] along line segment, or None if no intersection
        """
        # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm/1084899%231084899
        a = self.dot(d, d)
        b = 2 * self.dot(f, d)
        c = self.dot(f, f) - r * r
        discriminant = (b * b) - (4 * a * c)
        if discriminant >= 0:
            if use_t1:
                t1 = (-b - math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t1 <= 1:
                    return t1
            else:
                t2 = (-b + math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t2 <= 1:
                    return t2
        return None

    def detect_objects_from_ground_truth(self, obs, select_object_id=-1):
        """Detect objects using ground truth positions from MuJoCo simulation.
        
        Looks for cube1_pos, cube2_pos, cube3_pos in observations and selects
        target object based on selection criteria (smallest x or specific ID).
        
        Args:
            obs: Dictionary containing observation data with cube positions
            select_object_id: Object ID to select (1-3), or -1 to select smallest x
            
        Returns:
            List containing 3D position of selected object, or empty list if none found
        """
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
            if select_object_id == -1:
                cubes.sort(key=lambda x: x[0][0])  # Sort by x coordinate (first element of position)
                target_cube_pos, target_cube_id = cubes[0]
                detected_objects.append(target_cube_pos)
                print(f"Selected cube {target_cube_id} with smallest x value: {target_cube_pos[0]:.3f}")

            else:
                target_cube_pos, target_cube_id = cubes[select_object_id]
                detected_objects.append(target_cube_pos)
                print(f"Selected cube {target_cube_id} based on user defined id")
        return detected_objects



    def search_for_objects(self, obs):
        """Generate search action when no objects are detected.
        
        Rotates robot in place to look around for objects.
        
        Args:
            obs: Dictionary containing current robot state
            
        Returns:
            Action dictionary with rotation command and elevated arm pose
        """
        base_pose = obs['base_pose']
        
        # Simple search pattern: rotate in place to look around
        search_action = {
            'base_pose': np.array([base_pose[0], base_pose[1], base_pose[2] + 0.1]),
            'arm_pos': np.array([0.45, 0.0, 0.25]),  # Elevated position for searching
            'arm_quat': np.array([0.0, 0.707, 0.0, 0.707]),
            'gripper_pos': obs['gripper_pos'].copy(),
        }
        return search_action

    def distance(self, pt1, pt2):
        """Calculate Euclidean distance between two 2D points.
        
        Args:
            pt1: First point [x, y]
            pt2: Second point [x, y]
            
        Returns:
            Euclidean distance
        """
        return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

    def restrict_heading_range(self, h):
        """Normalize heading angle to [-π, π] range.
        
        Args:
            h: Heading angle in radians
            
        Returns:
            Normalized heading in [-π, π]
        """
        return (h + math.pi) % (2 * math.pi) - math.pi

    def get_end_effector_offset(self, primitive_name):
        """Calculate end-effector offset distance based on task type.
        
        This is the distance from base center to end effector when positioned
        for the given task. Used to compute base waypoints.
        
        Args:
            primitive_name: Task name ('pick', 'place', 'toss', 'shelf', 'drawer')
            
        Returns:
            Offset distance in meters
        """
        # Simplified version - assume gripper starts open
        gripper_open = True  
        if gripper_open:
            return 0.55 + 0.12
        return {'toss': 1.30, 'shelf': 0.75, 'drawer': 0.80}.get(primitive_name, 0.55)

    def build_base_command(self, command):
        """Build base movement command from high-level task command.
        
        Modifies waypoints so that when base reaches final waypoint, the end effector
        will be at the target position. Computes new waypoint based on end effector
        offset distance. Handles case where base is too close and needs to back up.
        
        Args:
            command: Dictionary with 'primitive_name' and 'waypoints' list
            
        Returns:
            Dictionary with 'waypoints' and 'target_ee_pos', or None if invalid command
        """
        assert command['primitive_name'] in {'move', 'pick', 'place', 'toss', 'shelf', 'drawer', 'push'}

        # Base movement only
        if command['primitive_name'] == 'move':
            return {'waypoints': command['waypoints'], 'target_ee_pos': None, 'position_tolerance': 0.1}

        # For push operation, base should be on reverse extension line
        # waypoints already set correctly, target_ee_pos should be object position
        if command['primitive_name'] == 'push':
            target_ee_pos = command['object_3d_pos'][:2].tolist()  # object position as target EE position
            return {
                'waypoints': command['waypoints'], 
                'target_ee_pos': target_ee_pos,
                'push_angle': command.get('push_angle'),
                'push_direction': command.get('push_direction')
            }

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
            print('Warning: Base needs to deviate from commanded path to reach target position, watch out for potential collisions')
            curr_position = command['waypoints'][0]
            signed_dist = self.distance(curr_position, target_ee_pos) - end_effector_offset
            dx = target_ee_pos[0] - curr_position[0]
            dy = target_ee_pos[1] - curr_position[1]
            target_heading = self.restrict_heading_range(math.atan2(dy, dx))
            target_position = (curr_position[0] + signed_dist * math.cos(target_heading), curr_position[1] + signed_dist * math.sin(target_heading))
            waypoints = [curr_position, target_position]
            
        return {'waypoints': waypoints, 'target_ee_pos': target_ee_pos}


if __name__ == '__main__':
    # WebServer(Queue()).run(); time.sleep(1000)
    # WebXRListener(); time.sleep(1000)
    from constants import POLICY_CONTROL_PERIOD
    obs = {
        'base_pose': np.zeros(3),
        'arm_pos': np.zeros(3),
        'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
        'gripper_pos': np.zeros(1),
        'base_image': np.zeros((640, 360, 3)),
        'wrist_image': np.zeros((640, 480, 3)),
    }
    policy = TeleopPolicy()
    # policy = RemotePolicy()
    while True:
        policy.reset()
        for _ in range(100):
            print(policy.step(obs))
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
