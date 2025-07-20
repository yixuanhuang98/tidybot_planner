import math
import time
import threading
import os
from datetime import datetime
from typing import Dict, Optional, Any
import numpy as np
import mujoco
import mujoco.viewer
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from constants import POLICY_CONTROL_PERIOD
from agent.ik_solver import IKSolver
from env.state import EnvironmentState, RobotState, ObjectState, Observation, Action
from env.mujoco.renderer import MultiViewRenderer


class BaseController:
    """Base controller for mobile robot base"""
    
    def __init__(self, qpos, qvel, ctrl, timestep):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl

        # OTG (online trajectory generation)
        num_dofs = 3
        self.last_command_time = time.time()
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = [0.5, 0.5, 3.14]
        self.otg_inp.max_acceleration = [0.5, 0.5, 2.36]
        self.otg_res = None

    def reset(self):
        """Reset controller to initial state"""
        self.qpos[:] = np.zeros(3)
        self.ctrl[:] = self.qpos

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos.copy()
        self.otg_inp.current_velocity = self.qvel.copy()
        self.otg_inp.target_position = self.qpos.copy()
        self.otg_res = Result.Finished

    def update(self, target_pose: Optional[np.ndarray] = None):
        """Update controller with new target pose"""
        if target_pose is not None:
            self.last_command_time = time.time()
            self.otg_inp.target_position = target_pose
            self.otg_res = Result.Working

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.qpos.copy()
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position


class ArmController:
    """Arm controller with inverse kinematics"""
    
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
        self.last_command_time = time.time()
        self.otg = Ruckig(num_dofs, timestep)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
        self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
        self.otg_res = None

    def reset(self):
        """Reset controller to initial state"""
        # Initialize arm in "retract" configuration
        self.qpos[:] = np.array([0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633])
        self.ctrl[:] = self.qpos.copy()
        self.ctrl_gripper[:] = 0.0

        # Initialize OTG
        self.last_command_time = time.time()
        self.otg_inp.current_position = self.qpos.copy()
        self.otg_inp.current_velocity = self.qvel.copy()
        self.otg_inp.target_position = self.qpos.copy()
        self.otg_res = Result.Finished

    def update(self, target_pos: Optional[np.ndarray] = None, 
               target_quat: Optional[np.ndarray] = None, 
               target_gripper: Optional[float] = None):
        """Update controller with new targets"""
        if target_pos is not None and target_quat is not None:
            self.last_command_time = time.time()
            
            # Run inverse kinematics
            qpos = self.ik_solver.solve(target_pos, target_quat, self.qpos)
            qpos = self.qpos + np.mod((qpos - self.qpos) + np.pi, 2 * np.pi) - np.pi
            
            self.otg_inp.target_position = qpos
            self.otg_res = Result.Working

        if target_gripper is not None:
            self.ctrl_gripper[:] = 255.0 * target_gripper

        # Maintain current pose if command stream is disrupted
        if time.time() - self.last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
            self.otg_inp.target_position = self.otg_out.new_position
            self.otg_res = Result.Working

        # Update OTG
        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.ctrl[:] = self.otg_out.new_position


class MujocoHandler:
    """MuJoCo simulation handler"""
    
    def __init__(self, mjcf_path: str = "env/assets/stanford_tidybot/scene.xml", 
                 show_viewer: bool = False, render_images: bool = False,
                 render_every_n_frames: int = 1):
        self.mjcf_path = mjcf_path
        self.show_viewer = show_viewer
        self.render_images = render_images
        self.render_every_n_frames = render_every_n_frames
        
        # MuJoCo simulation
        self.model = None
        self.data = None
        self.viewer = None
        self.viewer_thread = None
        
        # Controllers
        self.base_controller = None
        self.arm_controller = None
        
        # State
        self.current_state = EnvironmentState.create_empty()
        self.current_action = Action()
        self.lock = threading.RLock()
        
        # Simulation references
        self.qpos_base = None
        self.qpos_arm = None
        self.qpos_gripper = None
        self.qpos_cubes = {}
        
        # Site references for end effector
        self.site_xpos = None
        self.site_xmat = None
        self.site_quat = np.empty(4)
        self.base_height = 0.0
        self.base_rot_axis = np.array([0.0, 0.0, 1.0])
        self.base_quat_inv = np.empty(4)
        
        # Door site IDs
        self.door_site_ids = {}
        
        # Image saving setup
        self.image_step_counter = 0
        self.image_save_dir = None
        self.multi_view_renderer = None
        if self.render_images:
            self._setup_image_saving()

    def _setup_image_saving(self):
        """Setup directory for saving images"""
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.image_save_dir = f"simulation_images/{timestamp}"
        os.makedirs(self.image_save_dir, exist_ok=True)
        print(f"Images will be saved to: {self.image_save_dir}")
    
    def _setup_renderer(self):
        """Setup the multi-view renderer"""
        if self.render_images and self.model and self.data:
            try:
                self.multi_view_renderer = MultiViewRenderer(self.model, self.data)
                print("Multi-view renderer initialized successfully")
            except Exception as e:
                print(f"Failed to initialize multi-view renderer: {e}")
                self.multi_view_renderer = None

    def launch(self):
        """Launch the MuJoCo simulation"""
        # Load model
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        
        # Enable gravity compensation for everything except objects
        self.model.body_gravcomp[:] = 1.0
        body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        for object_name in ['cube1', 'cube2', 'cube3']:
            if object_name in body_names:
                self.model.body_gravcomp[self.model.body(object_name).id] = 0.0
        
        # Set up references to simulation data
        self._setup_simulation_references()
        
        # Initialize controllers
        self._setup_controllers()
        
        # Find special sites
        self._find_special_sites()
        
        # Set control callback
        mujoco.set_mjcb_control(self._control_callback)
        
        # Launch viewer if requested
        if self.show_viewer:
            self.viewer_thread = threading.Thread(target=self._launch_viewer, daemon=True)
            self.viewer_thread.start()
        
        # Setup renderer for image saving
        if self.render_images:
            self._setup_renderer()
        
        print("MuJoCo simulation launched successfully")

    def _setup_simulation_references(self):
        """Set up references to simulation data arrays"""
        base_dofs = self.model.body('base_link').jntnum.item()
        arm_dofs = 7
        
        # Robot joint references
        self.qpos_base = self.data.qpos[:base_dofs]
        qvel_base = self.data.qvel[:base_dofs]
        ctrl_base = self.data.ctrl[:base_dofs]
        
        qpos_arm = self.data.qpos[base_dofs:(base_dofs + arm_dofs)]
        qvel_arm = self.data.qvel[base_dofs:(base_dofs + arm_dofs)]
        ctrl_arm = self.data.ctrl[base_dofs:(base_dofs + arm_dofs)]
        
        self.qpos_gripper = self.data.qpos[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        ctrl_gripper = self.data.ctrl[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]
        
        # Object references
        cube_offset = base_dofs + arm_dofs + 8  # 8 for gripper joints
        self.qpos_cubes = {
            'cube1': self.data.qpos[cube_offset:cube_offset + 7],
            'cube2': self.data.qpos[cube_offset + 7:cube_offset + 14],
            'cube3': self.data.qpos[cube_offset + 14:cube_offset + 21]
        }
        
        # Store controller references
        self.base_ctrl_refs = (self.qpos_base, qvel_base, ctrl_base)
        self.arm_ctrl_refs = (qpos_arm, qvel_arm, ctrl_arm, self.qpos_gripper, ctrl_gripper)
        
        # End effector site
        site_id = self.model.site('pinch_site').id
        self.site_xpos = self.data.site(site_id).xpos
        self.site_xmat = self.data.site(site_id).xmat
        self.base_height = self.model.body('gen3/base_link').pos[2]

    def _setup_controllers(self):
        """Initialize robot controllers"""
        base_qpos, base_qvel, base_ctrl = self.base_ctrl_refs
        arm_qpos, arm_qvel, arm_ctrl, gripper_qpos, gripper_ctrl = self.arm_ctrl_refs
        
        self.base_controller = BaseController(base_qpos, base_qvel, base_ctrl, self.model.opt.timestep)
        self.arm_controller = ArmController(arm_qpos, arm_qvel, arm_ctrl, gripper_qpos, gripper_ctrl, self.model.opt.timestep)

    def _find_special_sites(self):
        """Find special sites like doors"""
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name in ['leftdoor_site', 'rightdoor_site']:
                self.door_site_ids[name] = i

    def _launch_viewer(self):
        """Launch MuJoCo viewer in separate thread"""
        mujoco.viewer.launch(self.model, self.data, show_left_ui=False, show_right_ui=False)

    def _control_callback(self, *_):
        """Control callback called by MuJoCo"""
        # Update controllers
        with self.lock:
            self.base_controller.update(self.current_action.base_pose)
            self.arm_controller.update(self.current_action.arm_pos, 
                                    self.current_action.arm_quat, 
                                    self.current_action.gripper_pos)
            
            # Update state
            self._update_state()

    def _update_state(self):
        """Update current state from simulation"""
        # Update robot state
        robot_state = self._get_robot_state()
        
        # Update object states
        object_states = self._get_object_states()
        
        # Update scene objects
        scene_objects = self._get_scene_objects()
        
        # Create new environment state
        self.current_state = EnvironmentState(
            robot=robot_state,
            objects=object_states,
            scene_objects=scene_objects,
            timestamp=time.time()
        )

    def _get_robot_state(self) -> RobotState:
        """Get current robot state"""
        # Base pose
        base_pose = self.qpos_base.copy()
        
        # Arm position and orientation in base frame
        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height
        site_xpos[:2] -= self.qpos_base[:2]
        mujoco.mju_axisAngle2Quat(self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2])
        arm_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(arm_pos, site_xpos, self.base_quat_inv)
        
        # Arm quaternion in base frame
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
        arm_quat = np.zeros(4)
        mujoco.mju_mulQuat(arm_quat, self.base_quat_inv, self.site_quat)
        arm_quat = arm_quat[[1, 2, 3, 0]]  # Convert to (x, y, z, w)
        if arm_quat[3] < 0.0:
            arm_quat *= -1
        
        # Gripper position
        gripper_pos = self.qpos_gripper[0] / 0.8
        
        return RobotState(
            base_pose=base_pose,
            arm_pos=arm_pos,
            arm_quat=arm_quat,
            gripper_pos=gripper_pos
        )

    def _get_object_states(self) -> Dict[str, ObjectState]:
        """Get current object states"""
        object_states = {}
        
        for obj_name, qpos in self.qpos_cubes.items():
            pos = qpos[:3].copy()
            quat = qpos[3:7].copy()
            quat = quat[[1, 2, 3, 0]]  # Convert to (x, y, z, w)
            if quat[3] < 0.0:
                quat *= -1
            
            object_states[obj_name] = ObjectState(pos=pos, quat=quat)
        
        return object_states

    def _get_scene_objects(self) -> Dict[str, Any]:
        """Get scene object states"""
        scene_objects = {}
        
        for site_name, site_id in self.door_site_ids.items():
            pos_key = site_name.replace('_site', '_pos')
            scene_objects[pos_key] = self.data.site_xpos[site_id].copy()
        
        return scene_objects

    def set_states(self, action: Optional[Action] = None):
        """Set states or apply action"""
        with self.lock:
            if action is None:
                # Reset simulation
                self._reset_simulation()
            else:
                # Set action
                self.current_action = action

    def _reset_simulation(self):
        """Reset the simulation"""
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)
        
        # Do not randomize objects if they are on the table or in cupboard mode
        if 'table' in self.mjcf_path or 'cupboard' in self.mjcf_path:
            print("Table or cupboard scene detected, skipping object randomization.")
        else:
            # Randomize cube positions
            for i, (obj_name, qpos) in enumerate(self.qpos_cubes.items()):
                # Randomize position
                qpos[:2] += np.random.uniform(-0.3, 0.3, 2)
                
                # Randomize orientation
                theta = np.random.uniform(-math.pi, math.pi)
                qpos[3:7] = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
                
                print(f"{obj_name} reset to position: [{qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f}], theta: {theta:.3f}")
        
        # Forward simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()
        
        # Clear current action
        self.current_action = Action()

    def step(self):
        """Step the simulation"""
        if self.model is not None:
            mujoco.mj_step(self.model, self.data)

    def get_states(self) -> Dict[str, Any]:
        """Get current states as dictionary"""
        with self.lock:
            return {
                'base_pose': self.current_state.robot.base_pose,
                'arm_pos': self.current_state.robot.arm_pos,
                'arm_quat': self.current_state.robot.arm_quat,
                'gripper_pos': self.current_state.robot.gripper_pos,
                **{f'{obj_name}_pos': obj_state.pos for obj_name, obj_state in self.current_state.objects.items()},
                **{f'{obj_name}_quat': obj_state.quat for obj_name, obj_state in self.current_state.objects.items()},
                **self.current_state.scene_objects
            }

    def get_extra(self) -> Dict[str, Any]:
        """Get extra information"""
        extra = {'timestamp': self.current_state.timestamp}
        
        # Add images if rendering is enabled
        if self.render_images:
            images = self._render_and_save_images()
            extra['images'] = images
        
        return extra

    def render(self):
        """Render the environment"""
        if self.render_images:
            return self._render_and_save_images()
        return None
    
    def _render_and_save_images(self):
        """Render images from multiple camera views and save them"""
        if not self.multi_view_renderer:
            return {}
        
        # Check if we should render this frame
        should_render = (self.image_step_counter % self.render_every_n_frames) == 0
        
        # Increment counter first
        self.image_step_counter += 1
        
        # Only render and save if it's time to do so
        if not should_render:
            return {}
        
        try:
            # Render all views
            images = self.multi_view_renderer.render_all_views()
            
            # Save images to disk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            
            for view_name, image in images.items():
                if image is not None:
                    filename = f"{view_name}_{timestamp}_{self.image_step_counter:06d}.jpg"
                    filepath = os.path.join(self.image_save_dir, filename)
                    self._save_image(image, filepath)
            
            return images
            
        except Exception as e:
            print(f"Error rendering and saving images: {e}")
            return {}
    
    
    def _save_image(self, image, filepath):
        """Save image to disk"""
        try:
            # Try to import PIL for better image saving
            from PIL import Image
            img = Image.fromarray(image)
            img.save(filepath, quality=95)
        except ImportError:
            # Fallback to matplotlib if PIL not available
            try:
                import matplotlib.pyplot as plt
                plt.imsave(filepath, image)
            except ImportError:
                # Fallback to basic numpy save
                np.save(filepath.replace('.jpg', '.npy'), image)
                print(f"Saved image as numpy array: {filepath.replace('.jpg', '.npy')}")
        except Exception as e:
            print(f"Error saving image {filepath}: {e}")

    def dict_to_array(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """Convert action dictionary to numpy array"""
        # Extract components from action dictionary
        base_pose = action_dict.get('base_pose', np.zeros(3))
        arm_pos = action_dict.get('arm_pos', np.zeros(3))
        arm_quat = action_dict.get('arm_quat', np.array([1.0, 0.0, 0.0, 0.0]))
        gripper_pos = action_dict.get('gripper_pos', np.array([0.0]))
        
        # Handle different input formats
        if isinstance(base_pose, list):
            base_pose = np.array(base_pose)
        if isinstance(arm_pos, list):
            arm_pos = np.array(arm_pos)
        if isinstance(arm_quat, list):
            arm_quat = np.array(arm_quat)
        if isinstance(gripper_pos, (int, float)):
            gripper_pos = np.array([gripper_pos])
        elif isinstance(gripper_pos, list):
            gripper_pos = np.array(gripper_pos)
        
        # Ensure proper shapes
        base_pose = base_pose.flatten()[:3]
        arm_pos = arm_pos.flatten()[:3]
        arm_quat = arm_quat.flatten()[:4]
        gripper_pos = gripper_pos.flatten()[:1]
        
        # Concatenate into single array
        return np.concatenate([base_pose, arm_pos, arm_quat, gripper_pos])

    def close(self):
        """Close the simulation"""
        if self.viewer_thread and self.viewer_thread.is_alive():
            self.viewer_thread.join(timeout=1.0)
        
        # Close multi-view renderer
        if self.multi_view_renderer:
            self.multi_view_renderer.close()
            self.multi_view_renderer = None
        
        if self.model is not None:
            # Clean up MuJoCo resources
            self.model = None
            self.data = None
        
        print("MuJoCo simulation closed")