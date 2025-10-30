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
import cv2 as cv
import mujoco
import mujoco.viewer
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from constants import POLICY_CONTROL_PERIOD
from ik_solver import IKSolver

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

# Adapted from https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/renderer.py
class Renderer:
    def __init__(self, model, data, shm_image):
        self.model = model
        self.data = data
        self.image = np.empty_like(shm_image.data)

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
        self.gl_context = mujoco.gl_context.GLContext(width, height)
        self.gl_context.make_current()
        self.mjr_context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.mjr_context)

        # Set up scene
        self.scene_option = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, 10000)

    def render(self):
        self.gl_context.make_current()
        mujoco.mjv_updateScene(self.model, self.data, self.scene_option, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(self.rect, self.scene, self.mjr_context)
        mujoco.mjr_readPixels(self.image, None, self.rect, self.mjr_context)
        self.shm_image.data[:] = np.flipud(self.image)

    def close(self):
        self.gl_context.free()
        self.gl_context = None
        self.mjr_context.free()
        self.mjr_context = None

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

    def reset(self, base_reset_pose=None):
        # Initialize base at origin
        if base_reset_pose is None:
            self.qpos[:] = np.zeros(3)
            self.qpos[:1] = np.random.uniform(-1, 1)
            self.qpos[2] = np.random.uniform(-np.pi, np.pi)
        else:
            self.qpos[:] = base_reset_pose
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
            # cube_qpos[0] += np.random.uniform(-0.1, 0.1, 1)  # X position
            # cube_qpos[1] += np.random.uniform(-0.2, 0.2, 1)  # Y position

            cube_qpos[:2] = np.random.uniform(-0.7, 0.7, 2)
            # Keep Z position at table height (don't randomize vertical position)
            
            # Randomize orientation around Z-axis (yaw)
            theta = np.random.uniform(-math.pi, math.pi)
            cube_qpos[3:7] = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
            
            print(f"Cube {i+1} reset to position: [{cube_qpos[0]:.3f}, {cube_qpos[1]:.3f}, {cube_qpos[2]:.3f}], theta: {theta:.3f}")
        
        mujoco.mj_forward(self.model, self.data)


        base_reset_pose = np.zeros(3)
        for _ in range(100):
            base_reset_pose[:1] = np.random.uniform(-1, 1)
            base_reset_pose[2] = np.random.uniform(-np.pi, np.pi)
            is_valid = True
            for cube_qpos in cubes:
                if np.linalg.norm(cube_qpos[:2] - base_reset_pose[:2]) < 0.35:
                    is_valid = False
                    break
            print(f"Cube 1 pos: [{cubes[0][0]:.3f}, {cubes[0][1]:.3f}], Cube 2 pos: [{cubes[1][0]:.3f}, {cubes[1][1]:.3f}], Cube 3 pos: [{cubes[2][0]:.3f}, {cubes[2][1]:.3f}]")
            print(f"Base reset pose: [{base_reset_pose[0]:.3f}, {base_reset_pose[1]:.3f}, {base_reset_pose[2]:.3f}], is_valid: {is_valid}")
            if is_valid:
                break
        # Reset controllers
        self.base_controller.reset(base_reset_pose)
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

class MujocoEnv:
    def __init__(self, render_images=True, show_viewer=True, show_images=True):
        self.mjcf_path = 'models/stanford_tidybot/ground_scene.xml'
        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images
        self.command_queue = mp.Queue(1)

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
        # Set up renderers
        renderers = [Renderer(model, data, shm_image) for shm_image in self.shm_images]

        # Render camera images continuously
        while True:
            start_time = time.time()
            for renderer in renderers:
                renderer.render()
            render_time = time.time() - start_time
            if render_time > 0.1:  # 10 fps
                print(f'Warning: Offscreen rendering took {1000 * render_time:.1f} ms, try making the Mujoco viewer window smaller to speed up offscreen rendering')

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
    env = MujocoEnv()
    # env = MujocoEnv(show_images=True)
    # env = MujocoEnv(render_images=False)
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
