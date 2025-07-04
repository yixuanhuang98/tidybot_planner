import time
from typing import Dict, Any, Optional
import cv2 as cv
import zmq

from agent.teleop_policy import TeleopPolicy
from constants import POLICY_SERVER_HOST, POLICY_SERVER_PORT, POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT


class RemotePolicy(TeleopPolicy):
    """Execute policy running on remote server with phone as enabling device"""
    
    def __init__(self):
        super().__init__()
        
        # Use phone as enabling device during policy rollout
        self.enabled = False
        
        # Connection to policy server
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')
        print(f'Connected to policy server at {POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')

    def reset(self):
        """Reset remote policy and wait for user signal"""
        # Wait for user to signal that episode has started
        super().reset()  # Note: Comment out to run without phone
        
        # Check connection to policy server and reset policy
        default_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # Temporarily set 1000 ms timeout
        self.socket.send_pyobj({'reset': True})
        try:
            self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        except zmq.error.Again as e:
            raise Exception('Could not communicate with policy server') from e
        self.socket.setsockopt(zmq.RCVTIMEO, default_timeout)  # Put default timeout back
        
        # Disable policy execution until user presses on screen
        self.enabled = False  # Note: Set to True to run without phone

    def _step(self, obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute remote policy step"""
        # Return teleop command if episode has ended
        if self.episode_ended:
            return self.teleop_controller.step(obs)
        
        # Return no action if robot is not enabled
        if not self.enabled:
            return None
        
        # Encode images for policy server
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

    def _process_message(self, data: Dict[str, Any]):
        """Process WebXR message for enabling/disabling policy"""
        if self.episode_ended:
            # Run teleop controller if episode has ended
            self.teleop_controller.process_message(data)
        else:
            # Enable policy execution if user is pressing on screen
            self.enabled = 'teleop_mode' in data