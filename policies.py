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
    def reset(self):
        raise NotImplementedError

    def step(self, obs):
        raise NotImplementedError

class WebServer:
    def __init__(self, queue):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.queue = queue

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.socketio.on('message')
        def handle_message(data):
            # Send the timestamp back for RTT calculation (expected RTT on 5 GHz Wi-Fi is 7 ms)
            emit('echo', data['timestamp'])

            # Add data to queue for processing
            self.queue.put(data)

        # Reduce verbose Flask log output
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    def run(self):
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
    # WebXR: +x right, +y up, +z back; Robot: +x forward, +y left, +z up
    pos = np.array([-pos['z'], -pos['x'], pos['y']], dtype=np.float64)
    rot = R.from_quat([-quat['z'], -quat['x'], quat['y'], quat['w']])

    # Apply offset so that rotations are around device center instead of device camera
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)

    return pos, rot

TWO_PI = 2 * math.pi

class TeleopController:
    def __init__(self):
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
    def __init__(self, enable_web_server=True):
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
        # Signal that user has ended episode
        if not self.episode_ended and self.teleop_state == 'episode_ended':
            self.episode_ended = True
            return 'end_episode'

        # Signal that user is ready for env reset (after ending the episode)
        if self.teleop_state == 'reset_env':
            return 'reset_env'

        return self._step(obs)

    def _step(self, obs):
        return self.teleop_controller.step(obs)

    def listener_loop(self):
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
        self.teleop_controller.process_message(data)

# Execute policy running on remote server
class RemotePolicy(TeleopPolicy):
    def __init__(self, enable_web_server=True):
        super().__init__(enable_web_server=enable_web_server)

        # Use phone as enabling device during policy rollout
        self.enabled = False

        # Connection to policy server
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')
        print(f'Connected to policy server at {POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')

    def reset(self):
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
        if self.episode_ended:
            # Run teleop controller if episode has ended
            self.teleop_controller.process_message(data)
        else:
            # Enable policy execution if user is pressing on screen
            self.enabled = 'teleop_mode' in data

# Motion Planner generated plan. 
class MotionPlannerPolicy(Policy):
    def __init__(self):
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
        
        # Enable policy execution immediately (no web interface required)
        self.enabled = True
        self.episode_ended = False
        
        print(f'Motion planner policy initialized - ready to start automatically')

    def reset(self):
        # Reset motion planning state
        self.state = 'idle'
        self.current_command = None
        self.base_waypoints = []
        self.current_waypoint_idx = 0
        self.target_ee_pos = None
        self.lookahead_position = None
        self.episode_ended = False
        self.grasp_step = 0  # Reset grasping step
        
        # Clean up any grasp tracking variables
        if hasattr(self, 'grasp_start_time'):
            delattr(self, 'grasp_start_time')
        if hasattr(self, 'initial_gripper_pos'):
            delattr(self, 'initial_gripper_pos')
        
        # Enable policy execution immediately
        self.enabled = True
        
        print("Motion planner reset - starting episode automatically")

    def step(self, obs):
        # Return no action if episode has ended
        if self.episode_ended:
            return 'end_episode'

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

        # State machine following controller.py pattern
        if self.state == 'idle':
            # Detect objects and plan new command
            detected_objects = self.detect_objects_from_ground_truth(obs, select_object_id=1)
            if detected_objects:
                # Create pick command
                self.object_location = detected_objects[0]
                # Set placement location relative to detected object (e.g., 50cm away)
                self.target_location = np.array([
                    self.object_location[0] + 0.5,  # 50cm in X direction
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
                if base_command:
                    self.current_command = pick_command
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
            else:
                # No objects found, search around
                return self.search_for_objects(obs)
                
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
                        else:
                            print(f'Too far from target end effector position ({(100 * diff):.1f} cm)')
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
                    # Step 1: Position arm well above object with open gripper (safe approach)
                    action = {
                        'base_pose': base_pose.copy(),
                        'arm_pos': object_relative_pos,  # Position arm above object
                        'arm_quat': np.array([1.0, 0.0, 0.0, 0.0]),  # Inverted: gripper fingers pointing down toward ground
                        # 'arm_quat': np.array([0.71334776,  0.70182338, 0.03010257,  0.01993094]),
                        # 'arm_quat': np.array([0.707,  0., 0.707,  0.]),
                        # 'arm_quat': np.array([0.6204814, 0.3390396, 0.6206078, 0.3389705]),
                        'gripper_pos': np.array([0.0])  # Keep gripper open while positioning
                    }
                    print(f"Step 1: Positioning arm above object with open gripper")
                    # Move to next step after arm is positioned
                    if np.allclose(arm_pos_from_mount, object_relative_pos, atol=0.03):  # Tighter tolerance: 3cm
                        self.grasp_step = 1
                        print("Arm positioned above object, moving to lower approach")
                
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
            # Turn to face target end effector position
            dx = self.target_ee_pos[0] - base_pose[0]
            dy = self.target_ee_pos[1] - base_pose[1]
            desired_heading = math.atan2(dy, dx)  # Removed + math.pi to point towards target, not away
            
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
            target_heading += frac * heading_diff
            print(f"Heading: current={base_pose[2]:.3f}, desired={desired_heading:.3f}, diff={heading_diff:.3f}, frac={frac:.3f}, target={target_heading:.3f}")
        
        arm_pos_from_mount = obs['arm_pos'] - np.array([self.ARM_MOUNT_OFFSET[0], self.ARM_MOUNT_OFFSET[1], 0])
        # Create action to move towards target
        action = {
            'base_pose': np.array([target_position[0], target_position[1], target_heading]),
            'arm_pos': arm_pos_from_mount.copy(),
            'arm_quat': obs['arm_quat'].copy(),
            'gripper_pos': obs['gripper_pos'].copy(),
        }
        
        return action

    def dot(self, a, b):
        """Dot product helper function from controller.py"""
        return a[0] * b[0] + a[1] * b[1]

    def intersect(self, d, f, r, use_t1=False):
        """Line-circle intersection from controller.py"""
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
        """Detect objects using ground truth from MuJoCo simulation and find the one with smallest x value"""
        detected_objects = []
        
        # Get all three cube positions from MuJoCo environment
        cubes = []
        cube_keys = []
        for key in obs:
            print('key: ', key)
            if '0_pos' in key:
                cube_keys.append(key)
        for i in range(1, 4):
            # cube_key = f'cube{i}_pos'
            cube_key = cube_keys[i-1]
            print('obs keys: ', obs.keys())
            
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
        """Search behavior when no objects are detected"""
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
        """Calculate distance between two points from controller.py"""
        return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

    def restrict_heading_range(self, h):
        """Normalize heading to [-π, π] range from controller.py"""
        return (h + math.pi) % (2 * math.pi) - math.pi

    def get_end_effector_offset(self, primitive_name):
        """Calculate end-effector offset based on task and gripper state from controller.py"""
        # Simplified version - assume gripper starts open
        gripper_open = True  
        if gripper_open:
            return 0.55 + 0.12
        return {'toss': 1.30, 'shelf': 0.75, 'drawer': 0.80}.get(primitive_name, 0.55)

    def build_base_command(self, command):
        """Build base command using exact logic from controller.py"""
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
