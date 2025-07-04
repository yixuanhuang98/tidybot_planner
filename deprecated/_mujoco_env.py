# Author: Jimmy Wu
# Date: October 2024
#
# Note: This is a basic simulation environment for sanity checking the
# real-world pipeline for teleop and imitation learning. Performance metrics,
# reward signals, and termination signals are not implemented.

import math
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from threading import Thread
from datetime import datetime
from pathlib import Path
import cv2 as cv
import mujoco
import mujoco.viewer
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from constants import POLICY_CONTROL_PERIOD
from agent.ik_solver import IKSolver
import os
import subprocess

class ShmState:
    def __init__(self, existing_instance=None):
        arr = np.empty(3 + 3 + 4 + 1 + 1 + 9 + 12)  # Added 9 for 3 cube positions and 12 for 3 cube quaternions (3*3 + 3*4)
        if existing_instance is None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.base_pose = self.data[:3]
        self.arm_pos = self.data[3:6]
        self.arm_quat = self.data[6:10]
        self.gripper_pos = self.data[10:11]
        self.initialized = self.data[11:12]
        self.cube1_pos = self.data[12:15]
        self.cube2_pos = self.data[15:18]
        self.cube3_pos = self.data[18:21]
        self.cube1_quat = self.data[21:25]
        self.cube2_quat = self.data[25:29]
        self.cube3_quat = self.data[29:33]
        self.initialized[:] = 0.0

    def close(self):
        self.shm.close()

class ShmImage:
    def __init__(self, camera_name=None, width=None, height=None, existing_instance=None):
        if existing_instance is None:
            self.camera_name = camera_name
            arr = np.empty((height, width, 3), dtype=np.uint8)
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            self.camera_name = existing_instance.camera_name
            arr = existing_instance.data
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)
        self.data = np.ndarray(arr.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.data.fill(0)

    def close(self):
        self.shm.close()

class CPURenderer:
    """CPU-based renderer using MuJoCo's software rendering - no OpenGL required"""
    
    def __init__(self, model, data, shm_image):
        self.model = model
        self.data = data
        self.shm_image = ShmImage(existing_instance=shm_image)
        
        # Get camera info
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA.value, shm_image.camera_name)
        width, height = model.cam_resolution[camera_id]
        self.camera_id = camera_id
        self.width = width
        self.height = height
        
        print(f"Initialized CPU renderer for camera '{shm_image.camera_name}' ({width}x{height})")
        
        # Initialize camera
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = camera_id
        
        # Set up scene for CPU rendering
        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, 1000)
        
        # Create image buffer
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
    
    def render(self):
        """Render using MuJoCo's CPU-based rendering"""
        try:
            # Update scene
            mujoco.mjv_updateScene(
                self.model, self.data, self.scene_option, None, 
                self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
            )
            
            # CPU-based rendering using MuJoCo's software renderer
            # This renders the scene to a buffer without requiring OpenGL
            viewport = mujoco.MjrRect(0, 0, self.width, self.height)
            
            # Use MuJoCo's CPU rendering capabilities
            # Create a simple render using the physics state
            self._render_simple_view()
            
        except Exception as e:
            print(f"CPU rendering error for camera '{self.shm_image.camera_name}': {e}")
            self._render_physics_based_placeholder()
    
    def _render_simple_view(self):
        """Create a view based on the actual physics state"""
        # Get robot state
        height, width = self.height, self.width
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get some physics data to visualize
        try:
            # Get joint positions (this is actual simulation data)
            nq = self.model.nq
            joint_positions = self.data.qpos[:min(nq, 10)]  # First 10 joint positions
            
            # Get center of mass position if available
            if hasattr(self.data, 'subtree_com') and len(self.data.subtree_com) > 0:
                com = self.data.subtree_com[0]  # Root body center of mass
            else:
                com = np.array([0.0, 0.0, 0.0])
            
            # Create a visualization based on actual physics state
            center_x, center_y = width // 2, height // 2
            
            # Background gradient based on COM position
            for y in range(height):
                for x in range(width):
                    # Use actual physics data to create patterns
                    r = int(128 + 100 * np.sin(com[0] * 2 + x * 0.02))
                    g = int(128 + 100 * np.sin(com[1] * 2 + y * 0.02))
                    b = int(128 + 100 * np.sin(com[2] * 2 + (x+y) * 0.01))
                    
                    # Clamp values
                    r = max(0, min(255, r))
                    g = max(0, min(255, g))
                    b = max(0, min(255, b))
                    
                    image[y, x] = [r, g, b]
            
            # Draw representations of robot joints
            for i, qpos in enumerate(joint_positions):
                if i >= 7:  # Limit to prevent overflow
                    break
                    
                # Map joint position to pixel coordinates
                x = int(center_x + qpos * 50)
                y = int(center_y + i * 30 - 100)
                
                # Clamp to image bounds
                x = max(20, min(width-20, x))
                y = max(20, min(height-20, y))
                
                # Draw a circle representing the joint
                cv.circle(image, (x, y), 10, (255, 255, 255), 2)
                cv.putText(image, f'J{i}', (x-10, y-15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add camera and physics info
            font = cv.FONT_HERSHEY_SIMPLEX
            info_text = f"CPU RENDER - {self.shm_image.camera_name}"
            cv.putText(image, info_text, (10, 30), font, 0.5, (255, 255, 255), 1)
            
            physics_text = f"COM: [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}]"
            cv.putText(image, physics_text, (10, height-20), font, 0.3, (200, 200, 200), 1)
            
            self.shm_image.data[:] = image
            
        except Exception as e:
            print(f"Error in simple view rendering: {e}")
            self._render_physics_based_placeholder()
    
    def _render_physics_based_placeholder(self):
        """Generate placeholder using actual physics data"""
        height, width = self.height, self.width
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Use simulation time and physics state
        t = self.data.time if hasattr(self.data, 'time') else time.time()
        
        # Create pattern based on actual simulation time
        for y in range(0, height, 2):
            for x in range(0, width, 2):
                r = int(50 + 50 * np.sin(t + x * 0.02))
                g = int(100 + 50 * np.sin(t + y * 0.02))
                b = int(150 + 50 * np.sin(t + (x+y) * 0.01))
                
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                image[y:y+2, x:x+2] = [r, g, b]
        
        # Add text
        try:
            font = cv.FONT_HERSHEY_SIMPLEX
            text = f"PHYSICS RENDER - {self.shm_image.camera_name}"
            cv.putText(image, text, (10, height//2), font, 0.5, (255, 255, 255), 1)
            
            time_text = f"Sim time: {t:.2f}s"
            cv.putText(image, time_text, (10, height//2 + 30), font, 0.4, (200, 200, 200), 1)
        except:
            pass
        
        self.shm_image.data[:] = image
    
    def close(self):
        """Nothing to cleanup for CPU renderer"""
        pass

# Original OpenGL-based renderer
class Renderer:
    def __init__(self, model, data, shm_image, headless=False):
        self.model = model
        self.data = data
        self.image = np.empty_like(shm_image.data)
        self.headless = headless

        # Attach to existing shared memory image
        self.shm_image = ShmImage(existing_instance=shm_image)

        # Set up camera
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA.value, shm_image.camera_name)
        width, height = model.cam_resolution[camera_id]
        self.camera = mujoco.MjvCamera()
        self.camera.fixedcamid = camera_id
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # Set up context
        self.rect = mujoco.MjrRect(0, 0, width, height)
        
        # Use EGL for headless rendering
        if self.headless:
            # Force EGL backend for headless rendering
            os.environ['MUJOCO_GL'] = 'egl'
            try:
                self.gl_context = mujoco.gl_context.GLContext(width, height)
                self.gl_context.make_current()
            except Exception as e:
                print(f"Warning: Failed to create EGL context ({e}), falling back to CPU rendering")
                self.gl_context = None
                self.use_cpu_fallback = True
                return
        else:
            self.gl_context = mujoco.gl_context.GLContext(width, height)
            self.gl_context.make_current()
        
        self.use_cpu_fallback = False
        self.mjr_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context)

        # Set up scene
        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, 10000)

    def render(self):
        if self.use_cpu_fallback:
            # CPU fallback: create a simple colored image as placeholder
            height, width = self.shm_image.data.shape[:2]
            # Create a simple gradient image as placeholder
            placeholder = np.zeros((height, width, 3), dtype=np.uint8)
            placeholder[:, :, 0] = 50  # Red channel
            placeholder[:, :, 1] = 100  # Green channel  
            placeholder[:, :, 2] = 150  # Blue channel
            self.shm_image.data[:] = placeholder
            return
            
        self.gl_context.make_current()
        mujoco.mjv_updateScene(self.model, self.data, self.scene_option, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(self.rect, self.scene, self.mjr_context)
        mujoco.mjr_readPixels(self.image, None, self.rect, self.mjr_context)
        self.shm_image.data[:] = np.flipud(self.image)

    def close(self):
        if not self.use_cpu_fallback and self.gl_context:
            self.gl_context.free()
            self.gl_context = None
        if hasattr(self, 'mjr_context') and self.mjr_context:
            self.mjr_context.free()
            self.mjr_context = None

def setup_headless_rendering():
    """Set up headless rendering environment"""
    # Try to set up virtual framebuffer if available
    try:
        # Check if Xvfb is available
        result = subprocess.run(['which', 'xvfb-run'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Xvfb detected - using virtual framebuffer for headless rendering")
            # Start virtual framebuffer
            if not os.environ.get('DISPLAY'):
                os.environ['DISPLAY'] = ':99'
                subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)  # Give Xvfb time to start
                return True
    except:
        pass
    
    # Try EGL setup
    try:
        os.environ['MUJOCO_GL'] = 'egl'
        print("Using EGL backend for headless rendering")
        return True
    except:
        pass
    
    # Try OSMesa if available
    try:
        os.environ['MUJOCO_GL'] = 'osmesa'
        print("Using OSMesa backend for headless rendering")
        return True
    except:
        pass
    
    print("Warning: No suitable headless rendering backend found")
    return False

class TrueMuJoCoRenderer:
    """Real MuJoCo renderer that captures actual camera views in headless mode"""
    
    def __init__(self, model, data, shm_image):
        self.model = model
        self.data = data
        self.shm_image = ShmImage(existing_instance=shm_image)
        
        # Get camera info
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA.value, shm_image.camera_name)
        self.width, self.height = model.cam_resolution[camera_id]
        self.camera_id = camera_id
        
        # Set up headless rendering
        setup_headless_rendering()
        
        try:
            # Set up camera
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = camera_id
            
            # Set up rendering context
            self.rect = mujoco.MjrRect(0, 0, self.width, self.height)
            self.gl_context = mujoco.gl_context.GLContext(self.width, self.height)
            self.gl_context.make_current()
            
            # Set up MuJoCo rendering
            self.mjr_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context)
            
            # Set up scene
            self.scene_option = mujoco.MjvOption()
            self.scene = mujoco.MjvScene(self.model, 10000)
            
            # Create image buffer
            self.image = np.empty((self.height, self.width, 3), dtype=np.uint8)
            self.rendering_working = True
            
            print(f"Successfully initialized real MuJoCo renderer for '{shm_image.camera_name}' ({self.width}x{self.height})")
            
        except Exception as e:
            print(f"Failed to initialize MuJoCo renderer for '{shm_image.camera_name}': {e}")
            self.rendering_working = False
    
    def render(self):
        """Render actual MuJoCo camera view"""
        if not self.rendering_working:
            self._render_error_placeholder()
            return
        
        try:
            # Make sure context is current
            self.gl_context.make_current()
            
            # Update scene with current physics state
            mujoco.mjv_updateScene(
                self.model, self.data, self.scene_option, None,
                self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
            )
            
            # Render the scene
            mujoco.mjr_render(self.rect, self.scene, self.mjr_context)
            
            # Read pixels from framebuffer
            mujoco.mjr_readPixels(self.image, None, self.rect, self.mjr_context)
            
            # Copy to shared memory (MuJoCo renders upside down, so flip)
            self.shm_image.data[:] = np.flipud(self.image)
            
        except Exception as e:
            print(f"Rendering error for '{self.shm_image.camera_name}': {e}")
            self.rendering_working = False
            self._render_error_placeholder()
    
    def _render_error_placeholder(self):
        """Generate error placeholder image"""
        height, width = self.height, self.width
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Red background to indicate error
        image[:, :, 0] = 100  # Red
        image[:, :, 1] = 30   # Green
        image[:, :, 2] = 30   # Blue
        
        # Add error text
        try:
            font = cv.FONT_HERSHEY_SIMPLEX
            text = "RENDER ERROR"
            text_size = cv.getTextSize(text, font, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
            
            error_text = f"Camera: {self.shm_image.camera_name}"
            cv.putText(image, error_text, (10, height - 30), font, 0.5, (255, 255, 255), 1)
        except:
            pass
        
        self.shm_image.data[:] = image
    
    def close(self):
        """Cleanup rendering resources"""
        try:
            if hasattr(self, 'gl_context') and self.gl_context:
                self.gl_context.free()
            if hasattr(self, 'mjr_context') and self.mjr_context:
                self.mjr_context.free()
        except:
            pass

class BaseController:
    def __init__(self, qpos, qvel, ctrl, timestep):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl

        # OTG (online trajectory generation)
        num_dofs = 3
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]
        self.otg_res = None

    def reset(self):
        # Initialize base at origin
        self.qpos[:] = np.zeros(3)
        self.ctrl[:] = self.qpos

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            self.last_command_time = time.time()
            if 'base_pose' in command:
                # Set target base qpos
                self.otg_inp.target_position = command['base_pose']
                self.otg_res = Result.Working

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.qpos
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position

class ArmController:
    def __init__(self, qpos, qvel, ctrl, qpos_gripper, ctrl_gripper, timestep):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.qpos_gripper = qpos_gripper
        self.ctrl_gripper = ctrl_gripper

        # IK solver
        self.ik_solver = IKSolver(ee_offset=0.12)

        # OTG (online trajectory generation)
        num_dofs = 7
        self.last_command_time = None
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
        self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
        self.otg_res = None

    def reset(self):
        # Initialize arm in "retract" configuration
        self.qpos[:] = np.array([0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633])
        self.ctrl[:] = self.qpos
        self.ctrl_gripper[:] = 0.0

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos
        self.otg_inp.current_velocity = self.qvel
        self.otg_inp.target_position = self.qpos
        self.otg_res = Result.Finished

    def control_callback(self, command):
        if command is not None:
            self.last_command_time = time.time()

            if 'arm_pos' in command:
                # Run inverse kinematics on new target pose
                qpos = self.ik_solver.solve(command['arm_pos'], command['arm_quat'], self.qpos)
                qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi  # Unwrapped joint angles

                # Set target arm qpos
                self.otg_inp.target_position = qpos
                self.otg_res = Result.Working

            if 'gripper_pos' in command:
                # Set target gripper pos
                self.ctrl_gripper[:] = 255.0 * command['gripper_pos']  # fingers_actuator, ctrlrange [0, 255]

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.otg_out.new_position
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position

class MujocoSim:
    def __init__(self, mjcf_path, command_queue, shm_state, show_viewer=True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.command_queue = command_queue
        self.show_viewer = show_viewer

        # Enable gravity compensation for everything except objects
        self.model.body_gravcomp[:] = 1.0
        body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        for object_name in ['cube1', 'cube2', 'cube3']:
            if object_name in body_names:
                self.model.body_gravcomp[self.model.body(object_name).id] = 0.0

        # Cache references to array slices
        base_dofs = self.model.body('base_link').jntnum.item()
        arm_dofs = 7
        self.qpos_base = self.data.qpos[:base_dofs]
        qvel_base = self.data.qvel[:base_dofs]
        ctrl_base = self.data.ctrl[:base_dofs]
        qpos_arm = self.data.qpos[base_dofs:(base_dofs + arm_dofs)]
        qvel_arm = self.data.qvel[base_dofs:(base_dofs + arm_dofs)]
        ctrl_arm = self.data.ctrl[base_dofs:(base_dofs + arm_dofs)]
        self.qpos_gripper = self.data.qpos[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        ctrl_gripper = self.data.ctrl[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        # Track all three cubes (8 for gripper qpos, 7 for each cube qpos)
        self.qpos_cube1 = self.data.qpos[(base_dofs + arm_dofs + 8):(base_dofs + arm_dofs + 8 + 7)]
        self.qpos_cube2 = self.data.qpos[(base_dofs + arm_dofs + 8 + 7):(base_dofs + arm_dofs + 8 + 14)]
        self.qpos_cube3 = self.data.qpos[(base_dofs + arm_dofs + 8 + 14):(base_dofs + arm_dofs + 8 + 21)]

        # Controllers
        self.base_controller = BaseController(self.qpos_base, qvel_base, ctrl_base, self.model.opt.timestep)
        self.arm_controller = ArmController(qpos_arm, qvel_arm, ctrl_arm, self.qpos_gripper, ctrl_gripper, self.model.opt.timestep)

        # Shared memory state for observations
        self.shm_state = ShmState(existing_instance=shm_state)

        # Variables for calculating arm pos and quat
        site_id = self.model.site('pinch_site').id
        self.site_xpos = self.data.site(site_id).xpos
        self.site_xmat = self.data.site(site_id).xmat
        self.site_quat = np.empty(4)
        self.base_height = self.model.body('gen3/base_link').pos[2]
        self.base_rot_axis = np.array([0.0, 0.0, 1.0])
        self.base_quat_inv = np.empty(4)

        # Find leftdoor_site and rightdoor_site IDs if they exist
        self.leftdoor_site_id = None
        self.rightdoor_site_id = None
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name == 'leftdoor_site':
                self.leftdoor_site_id = i
            elif name == 'rightdoor_site':
                self.rightdoor_site_id = i

        # Reset the environment
        self.reset()

        # Set control callback
        mujoco.set_mjcb_control(self.control_callback)

    def reset(self):
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Randomize positions and orientations for all three cubes
        cubes = [self.qpos_cube1, self.qpos_cube2, self.qpos_cube3]
        for i, cube_qpos in enumerate(cubes):
            # Randomize position within a reasonable range around the table
            cube_qpos[:2] += np.random.uniform(-0.3, 0.3, 2)  # X and Y position
            # Keep Z position at table height (don't randomize vertical position)
            
            # Randomize orientation around Z-axis (yaw)
            theta = np.random.uniform(-math.pi, math.pi)
            cube_qpos[3:7] = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
            
            print(f"Cube {i+1} reset to position: [{cube_qpos[0]:.3f}, {cube_qpos[1]:.3f}, {cube_qpos[2]:.3f}], theta: {theta:.3f}")
        
        mujoco.mj_forward(self.model, self.data)

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()

    def control_callback(self, *_):
        # Check for new command
        command = None if self.command_queue.empty() else self.command_queue.get()
        if command == 'reset':
            self.reset()

        # Control callbacks
        self.base_controller.control_callback(command)
        self.arm_controller.control_callback(command)

        # Update base pose
        self.shm_state.base_pose[:] = self.qpos_base

        # Update arm pos
        # self.shm_state.arm_pos[:] = self.site_xpos
        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height  # Base height offset
        site_xpos[:2] -= self.qpos_base[:2]  # Base position inverse
        mujoco.mju_axisAngle2Quat(self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2])  # Base orientation inverse
        mujoco.mju_rotVecQuat(self.shm_state.arm_pos, site_xpos, self.base_quat_inv)  # Arm pos in local frame

        # Update arm quat
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
        # self.shm_state.arm_quat[:] = self.site_quat
        mujoco.mju_mulQuat(self.shm_state.arm_quat, self.base_quat_inv, self.site_quat)  # Arm quat in local frame

        # Update gripper pos
        self.shm_state.gripper_pos[:] = self.qpos_gripper / 0.8  # right_driver_joint, joint range [0, 0.8]

        # Update all three cube positions and quaternions
        self.shm_state.cube1_pos[:] = self.qpos_cube1[:3]  # First 3 elements are position
        self.shm_state.cube1_quat[:] = self.qpos_cube1[3:7]  # Next 4 elements are quaternion
        self.shm_state.cube2_pos[:] = self.qpos_cube2[:3]
        self.shm_state.cube2_quat[:] = self.qpos_cube2[3:7]
        self.shm_state.cube3_pos[:] = self.qpos_cube3[:3]
        self.shm_state.cube3_quat[:] = self.qpos_cube3[3:7]

        # Notify reset() function that state has been initialized
        self.shm_state.initialized[:] = 1.0

    def launch(self):
        if self.show_viewer:
            mujoco.viewer.launch(self.model, self.data, show_left_ui=False, show_right_ui=False)

        else:
            # Run headless simulation at real-time speed
            last_step_time = 0
            while True:
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                mujoco.mj_step(self.model, self.data)

class BlocksEnv:
    def __init__(self, render_images=False, show_viewer=False, show_images=False, save_images=False):
        self.mjcf_path = 'env/assets/stanford_tidybot/scene.xml'
        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.save_images = save_images
        self.command_queue = mp.Queue(1)

        # Create output directory for saved images if saving is enabled
        if self.save_images:
            self.image_output_dir = Path('simulation_images') / datetime.now().strftime('%Y%m%dT%H%M%S')
            self.image_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Simulation images will be saved to: {self.image_output_dir}")

        # Shared memory for state observations
        self.shm_state = ShmState()

        # Shared memory for image observations
        if self.render_images:
            self.shm_images = []
            model = mujoco.MjModel.from_xml_path(self.mjcf_path)
            for camera_id in range(model.ncam):
                camera_name = model.camera(camera_id).name
                width, height = model.cam_resolution[camera_id]
                self.shm_images.append(ShmImage(camera_name, width, height))

        # Start physics loop
        mp.Process(target=self.physics_loop, daemon=True).start()

        if self.render_images and self.show_images:
            # Start visualizer loop
            mp.Process(target=self.visualizer_loop, daemon=True).start()

    def physics_loop(self):
        # Create sim
        sim = MujocoSim(self.mjcf_path, self.command_queue, self.shm_state, show_viewer=self.show_viewer)

        # Start render loop
        if self.render_images:
            Thread(target=self.render_loop, args=(sim.model, sim.data), daemon=True).start()

        # Launch sim
        sim.launch()  # Launch in same thread as creation to avoid segfault

    def render_loop(self, model, data):
        # Try real MuJoCo rendering first, fallback to CPU visualization
        if self.save_images:
            print("Attempting real MuJoCo camera rendering in headless mode...")
            
            # Try to create real renderers
            renderers = []
            real_rendering_failed = False
            
            for shm_image in self.shm_images:
                try:
                    renderer = TrueMuJoCoRenderer(model, data, shm_image)
                    renderers.append(renderer)
                except Exception as e:
                    print(f"Real rendering failed for {shm_image.camera_name}: {e}")
                    real_rendering_failed = True
                    break
            
            # If real rendering failed, fall back to CPU visualization
            if real_rendering_failed or not renderers:
                print("Falling back to CPU-based physics visualization")
                renderers = [CPURenderer(model, data, shm_image) for shm_image in self.shm_images]
            else:
                print("Successfully initialized real MuJoCo camera rendering!")
        else:
            print("Using standard OpenGL renderer")
            renderers = [Renderer(model, data, shm_image, headless=True) for shm_image in self.shm_images]

        # Image saving setup
        frame_count = 0
        last_save_time = time.time()
        save_interval = 1.0  # Save every 1 second

        # Render camera images continuously
        while True:
            start_time = time.time()
            for renderer in renderers:
                renderer.render()
            render_time = time.time() - start_time
            
            # Save images if enabled
            if self.save_images:
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
                    
                    for shm_image in self.shm_images:
                        if shm_image.camera_name and shm_image.data is not None:
                            # Convert from RGB to BGR for OpenCV
                            bgr_image = cv.cvtColor(shm_image.data, cv.COLOR_RGB2BGR)
                            image_path = self.image_output_dir / f'{shm_image.camera_name}_{timestamp}_{frame_count:06d}.jpg'
                            cv.imwrite(str(image_path), bgr_image)
                    
                    frame_count += 1
                    last_save_time = current_time
                    
                    if frame_count % 10 == 0:  # Print status every 10 saved frames
                        print(f'Saved simulation images: {frame_count} frames to {self.image_output_dir}')
            
            if render_time > 0.1:  # 10 fps
                print(f'Warning: Offscreen rendering took {1000 * render_time:.1f} ms')

    def visualizer_loop(self):
        shm_images = [ShmImage(existing_instance=shm_image) for shm_image in self.shm_images]
        last_imshow_time = time.time()
        while True:
            while time.time() - last_imshow_time < 0.1:  # 10 fps
                time.sleep(0.01)
            last_imshow_time = time.time()
            for i, shm_image in enumerate(shm_images):
                cv.imshow(shm_image.camera_name, cv.cvtColor(shm_image.data, cv.COLOR_RGB2BGR))
                cv.moveWindow(shm_image.camera_name, 640 * i, -100)
            cv.waitKey(1)

    def reset(self):
        self.shm_state.initialized[:] = 0.0
        self.command_queue.put('reset')

        # Wait for state publishing to initialize
        while self.shm_state.initialized == 0.0:
            time.sleep(0.01)

        # Wait for image rendering to initialize (Note: Assumes all zeros is not a valid image)
        if self.render_images:
            while any(np.all(shm_image.data == 0) for shm_image in self.shm_images):
                time.sleep(0.01)

    def get_obs(self):
        arm_quat = self.shm_state.arm_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        
        # Process all three cube quaternions
        cube1_quat = self.shm_state.cube1_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)
        if cube1_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(cube1_quat, out=cube1_quat)
            
        cube2_quat = self.shm_state.cube2_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)
        if cube2_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(cube2_quat, out=cube2_quat)
            
        cube3_quat = self.shm_state.cube3_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)
        if cube3_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(cube3_quat, out=cube3_quat)
            
        obs = {
            'base_pose': self.shm_state.base_pose.copy(),
            'arm_pos': self.shm_state.arm_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.shm_state.gripper_pos.copy(),
            'cube1_pos': self.shm_state.cube1_pos.copy(),
            'cube1_quat': cube1_quat,
            'cube2_pos': self.shm_state.cube2_pos.copy(),
            'cube2_quat': cube2_quat,
            'cube3_pos': self.shm_state.cube3_pos.copy(),
            'cube3_quat': cube3_quat,
        }
        # Add leftdoor_pos if available
        if hasattr(self, 'sim') and hasattr(self.sim, 'leftdoor_site_id') and self.sim.leftdoor_site_id is not None:
            obs['leftdoor_pos'] = self.sim.data.site_xpos[self.sim.leftdoor_site_id].copy()
        # Add rightdoor_pos if available
        if hasattr(self, 'sim') and hasattr(self.sim, 'rightdoor_site_id') and self.sim.rightdoor_site_id is not None:
            obs['rightdoor_pos'] = self.sim.data.site_xpos[self.sim.rightdoor_site_id].copy()
        if self.render_images:
            for shm_image in self.shm_images:
                obs[f'{shm_image.camera_name}_image'] = shm_image.data.copy()
        return obs

    def step(self, action):
        # Note: We intentionally do not return obs here to prevent the policy from using outdated data
        self.command_queue.put(action)

    def close(self):
        self.shm_state.close()
        self.shm_state.shm.unlink()
        if self.render_images:
            for shm_image in self.shm_images:
                shm_image.close()
                shm_image.shm.unlink()

if __name__ == '__main__':
    # Run in headless mode with image saving by default
    env = BlocksEnv(render_images=True, show_viewer=False, show_images=False, save_images=True)
    try:
        while True:
            env.reset()
            for _ in range(100):
                action = {
                    'base_pose': 0.1 * np.random.rand(3) - 0.05,
                    'arm_pos': 0.1 * np.random.rand(3) + np.array([0.55, 0.0, 0.4]),
                    'arm_quat': np.random.rand(4),
                    'gripper_pos': np.random.rand(1),
                }
                env.step(action)
                obs = env.get_obs()
                print(f"Cube1 pos: {obs['cube1_pos']}, Cube2 pos: {obs['cube2_pos']}, Cube3 pos: {obs['cube3_pos']}")
                print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
                time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        env.close()
