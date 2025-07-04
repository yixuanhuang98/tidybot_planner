import math
import time
import threading
from queue import Queue
from typing import Dict, Any, Optional
import socket
import logging

import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from scipy.spatial.transform import Rotation as R

from agent.base_agent import BaseAgent
from constants import POLICY_CONTROL_PERIOD


class WebServer:
    """Web server for WebXR teleop interface"""
    
    def __init__(self, queue: Queue):
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
        print(f'Starting teleop server at {address}:5000')
        self.socketio.run(self.app, host='0.0.0.0')


class TeleopController:
    """Controller for WebXR-based teleoperation"""
    
    def __init__(self):
        # Device management
        self.primary_device_id = None
        self.secondary_device_id = None
        self.enabled_counts = {}

        # Robot state
        self.base_pose = None

        # Target poses
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
        self.arm_ref_base_pose = None
        self.gripper_ref_pos = None

    def process_message(self, data: Dict[str, Any]):
        """Process incoming WebXR message"""
        if not self.targets_initialized:
            return

        device_id = data['device_id']

        # Update enabled count
        self.enabled_counts[device_id] = (
            self.enabled_counts.get(device_id, 0) + 1 
            if 'teleop_mode' in data else 0
        )

        # Assign primary and secondary devices
        if self.enabled_counts[device_id] > 2:
            if self.primary_device_id is None and device_id != self.secondary_device_id:
                self.primary_device_id = device_id
            elif self.secondary_device_id is None and device_id != self.primary_device_id:
                self.secondary_device_id = device_id
        elif self.enabled_counts[device_id] == 0:
            if device_id == self.primary_device_id:
                self.primary_device_id = None
                self.base_xr_ref_pos = None
                self.arm_xr_ref_pos = None
            elif device_id == self.secondary_device_id:
                self.secondary_device_id = None
                self.base_xr_ref_pos = None

        # Process teleop commands
        if self.primary_device_id is not None and 'teleop_mode' in data:
            pos, rot = self.convert_webxr_pose(data['position'], data['orientation'])

            if data['teleop_mode'] == 'base' or device_id == self.secondary_device_id:
                self._process_base_teleop(pos, rot)
            elif data['teleop_mode'] == 'arm':
                self._process_arm_teleop(pos, rot, data.get('gripper_delta', 0.0))
        elif self.primary_device_id is None:
            # Update target pose if base is pushed while teleop is disabled
            self.base_target_pose = self.base_pose

    def _process_base_teleop(self, pos: np.ndarray, rot: R):
        """Process base teleoperation"""
        # Store reference poses
        if self.base_xr_ref_pos is None:
            self.base_ref_pose = self.base_pose.copy()
            self.base_xr_ref_pos = pos[:2]
            self.base_xr_ref_rot_inv = rot.inv()

        # Update position
        self.base_target_pose[:2] = self.base_ref_pose[:2] + (pos[:2] - self.base_xr_ref_pos)

        # Update orientation
        base_fwd_vec_rotated = (rot * self.base_xr_ref_rot_inv).apply([1.0, 0.0, 0.0])
        base_target_theta = self.base_ref_pose[2] + math.atan2(base_fwd_vec_rotated[1], base_fwd_vec_rotated[0])
        self.base_target_pose[2] += (base_target_theta - self.base_target_pose[2] + math.pi) % (2 * math.pi) - math.pi

    def _process_arm_teleop(self, pos: np.ndarray, rot: R, gripper_delta: float):
        """Process arm teleoperation"""
        # Store reference poses
        if self.arm_xr_ref_pos is None:
            self.arm_xr_ref_pos = pos
            self.arm_xr_ref_rot_inv = rot.inv()
            self.arm_ref_pos = self.arm_target_pos.copy()
            self.arm_ref_rot = self.arm_target_rot
            self.arm_ref_base_pose = self.base_pose.copy()
            self.gripper_ref_pos = self.gripper_target_pos

        # Coordinate frame transformations
        z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.base_pose[2])
        z_rot_inv = z_rot.inv()
        ref_z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2])

        # Update position
        pos_diff = pos - self.arm_xr_ref_pos
        pos_diff += ref_z_rot.apply(self.arm_ref_pos) - z_rot.apply(self.arm_ref_pos)
        pos_diff[:2] += self.arm_ref_base_pose[:2] - self.base_pose[:2]
        self.arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)

        # Update orientation
        self.arm_target_rot = (z_rot_inv * (rot * self.arm_xr_ref_rot_inv) * ref_z_rot) * self.arm_ref_rot

        # Update gripper
        self.gripper_target_pos = np.clip(self.gripper_ref_pos + gripper_delta, 0.0, 1.0)

    @staticmethod
    def convert_webxr_pose(pos: Dict[str, float], quat: Dict[str, float]) -> tuple:
        """Convert WebXR pose to robot coordinates"""
        device_camera_offset = np.array([0.0, 0.02, -0.04])  # iPhone 14 Pro
        
        # WebXR: +x right, +y up, +z back; Robot: +x forward, +y left, +z up
        pos_array = np.array([-pos['z'], -pos['x'], pos['y']], dtype=np.float64)
        rot = R.from_quat([-quat['z'], -quat['x'], quat['y'], quat['w']])

        # Apply offset so rotations are around device center
        pos_array = pos_array + rot.apply(device_camera_offset)

        return pos_array, rot

    def step(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate action from teleoperation"""
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

        # Create action
        arm_quat = self.arm_target_rot.as_quat()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            arm_quat = -arm_quat

        return {
            'base_pose': self.base_target_pose.copy(),
            'arm_pos': self.arm_target_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_target_pos.copy(),
        }


class TeleopPolicy(BaseAgent):
    """WebXR-based teleoperation policy"""
    
    def __init__(self):
        super().__init__()
        self.web_server_queue = Queue()
        self.teleop_controller = None
        self.teleop_state = None  # States: episode_started -> episode_ended -> reset_env
        self.episode_ended = False

        # Start web server
        server = WebServer(self.web_server_queue)
        threading.Thread(target=server.run, daemon=True).start()

        # Start listener thread
        threading.Thread(target=self._listener_loop, daemon=True).start()

    def reset(self):
        """Reset the teleop policy"""
        self.teleop_controller = TeleopController()
        self.episode_ended = False

        # Wait for user to signal episode start
        self.teleop_state = None
        while self.teleop_state != 'episode_started':
            time.sleep(0.01)

    def step(self, obs: Dict[str, Any]) -> Optional[str]:
        """Step the teleop policy"""
        # Check for episode end signal
        if not self.episode_ended and self.teleop_state == 'episode_ended':
            self.episode_ended = True
            return 'end_episode'

        # Check for reset signal
        if self.teleop_state == 'reset_env':
            return 'reset_env'

        return self._step(obs)

    def _step(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Internal step method"""
        return self.teleop_controller.step(obs)

    def _listener_loop(self):
        """Listen for WebXR messages"""
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

    def _process_message(self, data: Dict[str, Any]):
        """Process WebXR message"""
        if self.teleop_controller:
            self.teleop_controller.process_message(data)