# Author: Jimmy Wu
# Date: October 2024
#
# Customizable MuJoCo environment that supports:
# - Custom floor textures from object_assets/use_textures/
# - Custom objects from object_assets/robocasa_objs/ and object_assets/objects/
# - Dynamic object placement and randomization

import math
import os
import xml.etree.ElementTree as ET
import numpy as np
from mujoco_env import MujocoEnv, MujocoSim, ShmState
import mujoco


class CustomizableMujocoSim(MujocoSim):
    """Extended MujocoSim that handles variable numbers of custom objects."""

    def __init__(self, mjcf_path, command_queue, shm_state, object_names, show_viewer=True):
        self.custom_object_names = object_names
        self.num_custom_objects = len(object_names)

        # Store the custom shm_state before calling parent
        self.custom_shm_state = shm_state

        # Initialize model first to get DOF information
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.command_queue = command_queue
        self.show_viewer = show_viewer

        # Enable gravity compensation for everything except objects
        self.model.body_gravcomp[:] = 1.0
        body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        for i in range(self.num_custom_objects):
            obj_body_name = f'object_{i}'
            if obj_body_name in body_names:
                self.model.body_gravcomp[self.model.body(obj_body_name).id] = 0.0

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

        # Track custom object qpos (each object has 7 DOF: 3 pos + 4 quat, plus 8 for gripper)
        self.qpos_objects_start = base_dofs + arm_dofs + 8
        self.qpos_objects = []
        for i in range(self.num_custom_objects):
            obj_qpos_start = self.qpos_objects_start + i * 7
            obj_qpos_end = obj_qpos_start + 7
            self.qpos_objects.append(self.data.qpos[obj_qpos_start:obj_qpos_end])

        # Controllers
        from mujoco_env import BaseController, ArmController
        self.base_controller = BaseController(self.qpos_base, qvel_base, ctrl_base, self.model.opt.timestep)
        self.arm_controller = ArmController(qpos_arm, qvel_arm, ctrl_arm, self.qpos_gripper, ctrl_gripper, self.model.opt.timestep)

        # Shared memory state for observations
        from mujoco_env import ShmState
        self.shm_state = self.custom_shm_state

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

    def control_callback(self, *_):
        """Override to handle custom objects."""
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
        site_xpos = self.site_xpos.copy()
        site_xpos[2] -= self.base_height  # Base height offset
        site_xpos[:2] -= self.qpos_base[:2]  # Base position inverse
        mujoco.mju_axisAngle2Quat(self.base_quat_inv, self.base_rot_axis, -self.qpos_base[2])  # Base orientation inverse
        mujoco.mju_rotVecQuat(self.shm_state.arm_pos, site_xpos, self.base_quat_inv)  # Arm pos in local frame

        # Update arm quat
        mujoco.mju_mat2Quat(self.site_quat, self.site_xmat)
        mujoco.mju_mulQuat(self.shm_state.arm_quat, self.base_quat_inv, self.site_quat)  # Arm quat in local frame

        # Update gripper pos
        self.shm_state.gripper_pos[:] = self.qpos_gripper / 0.8  # right_driver_joint, joint range [0, 0.8]

        # Update all custom object positions and quaternions
        for i in range(self.num_custom_objects):
            self.shm_state.object_positions[i][:] = self.qpos_objects[i][:3]  # First 3 elements are position
            self.shm_state.object_quaternions[i][:] = self.qpos_objects[i][3:7]  # Next 4 elements are quaternion

        # Notify reset() function that state has been initialized
        self.shm_state.initialized[:] = 1.0

    def reset(self):
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # sample N objects without collisions
        for _ in range(100):
            # sample N objects without collisions
            # pos_x = np.random.uniform(0.4, 0.8, size=self.num_custom_objects)
            # pos_y = np.random.uniform(-0.3, 0.3, size=self.num_custom_objects)
            # if not self.check_collisions(pos_x, pos_y):
            #     break

            # large range and no base collision
            pos_x = np.random.uniform(-0.8, 0.8, size=self.num_custom_objects)
            pos_y = np.random.uniform(-0.8, 0.8, size=self.num_custom_objects)
            if not self.check_collisions(pos_x, pos_y) and not self.check_base_collision(pos_x, pos_y):
                break
                
        
        # Randomize positions and orientations for all objects
        for i in range(self.num_custom_objects):
            obj_qpos = self.qpos_objects[i]

            # Randomize position within a reasonable range around the table
            # obj_qpos[0] += np.random.uniform(-0.1, 0.1)  # X position
            # obj_qpos[1] += np.random.uniform(-0.2, 0.2)  # Y position
            # Keep Z position at table height (don't randomize vertical position)

            # randomize position
            obj_qpos[0] = pos_x[i]
            obj_qpos[1] = pos_y[i]
            # Randomize orientation around Z-axis (yaw)
            theta = np.random.uniform(-math.pi, math.pi)
            # theta = 0
            obj_qpos[3:7] = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])

            obj_name = self.custom_object_names[i] if i < len(self.custom_object_names) else f'object_{i}'
            print(f"Object {obj_name} reset to position: [{obj_qpos[0]:.3f}, {obj_qpos[1]:.3f}, {obj_qpos[2]:.3f}], theta: {theta:.3f}")

        mujoco.mj_forward(self.model, self.data)

        # Reset controllers
        self.base_controller.reset()
        self.arm_controller.reset()

    def check_collisions(self, pos_x, pos_y):
        for i in range(self.num_custom_objects):
            for j in range(i+1, self.num_custom_objects):
                if np.linalg.norm(pos_x[i] - pos_x[j]) < 0.1 and np.linalg.norm(pos_y[i] - pos_y[j]) < 0.1:
                    return True
        return False
    
    def check_base_collision(self, pos_x, pos_y):
        for i in range(self.num_custom_objects):
            if np.linalg.norm(pos_x[i] - self.qpos_base[0]) < 0.3 and np.linalg.norm(pos_y[i] - self.qpos_base[1]) < 0.3:
                return True
        return False
    
    def launch(self):
        """Launch the simulation."""
        if self.show_viewer:
            mujoco.viewer.launch(self.model, self.data, show_left_ui=False, show_right_ui=False)
        else:
            # Run headless simulation at real-time speed
            import time
            last_step_time = 0
            while True:
                while time.time() - last_step_time < self.model.opt.timestep:
                    time.sleep(0.0001)
                last_step_time = time.time()
                mujoco.mj_step(self.model, self.data)


class CustomizableShmState(ShmState):
    """Extended shared memory state for variable number of objects."""

    def __init__(self, num_objects, existing_instance=None):
        self.num_objects = num_objects
        # Allocate space for: base_pose(3) + arm_pos(3) + arm_quat(4) + gripper_pos(1) + initialized(1)
        # + num_objects * (pos(3) + quat(4))
        object_state_size = num_objects * (3 + 4)
        arr = np.empty(3 + 3 + 4 + 1 + 1 + object_state_size)

        if existing_instance is None:
            from multiprocessing import shared_memory
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        else:
            from multiprocessing import shared_memory
            self.shm = shared_memory.SharedMemory(name=existing_instance.shm.name)

        self.data = np.ndarray(arr.shape, buffer=self.shm.buf)
        self.base_pose = self.data[:3]
        self.arm_pos = self.data[3:6]
        self.arm_quat = self.data[6:10]
        self.gripper_pos = self.data[10:11]
        self.initialized = self.data[11:12]

        # Dynamically allocate object positions and quaternions
        self.object_positions = []
        self.object_quaternions = []
        offset = 12
        for i in range(num_objects):
            self.object_positions.append(self.data[offset:offset+3])
            self.object_quaternions.append(self.data[offset+3:offset+7])
            offset += 7

        self.initialized[:] = 0.0


class CustomizableMujocoEnv(MujocoEnv):
    """
    Customizable MuJoCo environment with support for:
    - Custom floor textures
    - Custom objects from asset directories
    - Dynamic scene generation

    Args:
        floor_texture: Path to floor texture PNG file (relative to object_assets/use_textures/)
                      or None for default texture
        objects: List of object specifications. Each can be:
                 - String: name of object from robocasa_objs/ (e.g., 'apple_1') or objects/ (e.g., 'apple.glb')
                 - Dict: {'name': str, 'pos': [x, y, z], 'scale': float}
        render_images: Whether to render camera images
        show_viewer: Whether to show the MuJoCo viewer
        show_images: Whether to show camera image windows
    """

    def __init__(self, floor_texture=None, objects=None, render_images=True, show_viewer=True, show_images=True):
        self.floor_texture = floor_texture
        self.objects_config = objects or []
        self.render_images = render_images
        self.show_viewer = show_viewer
        self.show_images = show_images

        # Generate custom scene XML
        self.mjcf_path = self._generate_scene_xml()
        print(f"Generated custom scene at: {self.mjcf_path}")

        # Initialize parent class components
        import multiprocessing as mp
        self.command_queue = mp.Queue(1)

        # Create custom shared memory for objects
        self.num_objects = len(self.objects_config)
        self.object_names = [self._get_object_name(obj) for obj in self.objects_config]
        self.shm_state = CustomizableShmState(self.num_objects)

        # Shared memory for image observations
        if self.render_images:
            from mujoco_env import ShmImage
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

    def _get_object_name(self, obj_spec):
        """Extract object name from object specification."""
        if isinstance(obj_spec, dict):
            return obj_spec['name']
        return obj_spec

    def _generate_scene_xml(self):
        """Generate MuJoCo scene XML with custom floor and objects."""
        # Get absolute path to project root (where this script is located)
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Create root element
        root = ET.Element('mujoco', model='customizable_tidybot_scene')

        # Include tidybot.xml using just the filename
        # Since we write our XML to the same directory as tidybot.xml,
        # all relative paths (including meshdir="../assets") work exactly the same
        ET.SubElement(root, 'include', file='tidybot.xml')

        # Statistics
        ET.SubElement(root, 'statistic', center='0.25 0 0.6', extent='1.0', meansize='0.05')

        # Visual settings
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'headlight', diffuse='0.6 0.6 0.6', ambient='0.1 0.1 0.1', specular='0 0 0')
        ET.SubElement(visual, 'rgba', haze='0.15 0.25 0.35 1')
        ET.SubElement(visual, 'global', azimuth='120', elevation='-20')

        # Assets
        asset = ET.SubElement(root, 'asset')
        ET.SubElement(asset, 'texture', type='skybox', builtin='gradient',
                     rgb1='0.3 0.5 0.7', rgb2='0 0 0', width='512', height='3072')

        # Floor texture
        if self.floor_texture:
            # Use absolute path for texture
            texture_path = os.path.join(project_root, f'object_assets/use_textures/{self.floor_texture}')
            ET.SubElement(asset, 'texture', type='2d', name='groundplane',
                         file=texture_path)
        else:
        # Default checker texture
            ET.SubElement(asset, 'texture', type='2d', name='groundplane',
                            builtin='checker', mark='edge', rgb1='0.2 0.3 0.4',
                            rgb2='0.1 0.2 0.3', markrgb='0.8 0.8 0.8',
                            width='300', height='300')

        ET.SubElement(asset, 'material', name='groundplane', texture='groundplane',
                     texuniform='true', texrepeat='5 5')

        # Load object assets
        self._add_object_assets(asset)

        # World body
        worldbody = ET.SubElement(root, 'worldbody')
        ET.SubElement(worldbody, 'light', pos='0 0 1.5', directional='true')
        ET.SubElement(worldbody, 'geom', name='floor', size='0 0 0.05',
                     type='plane', material='groundplane')

        # Add objects to scene
        self._add_objects_to_worldbody(worldbody)

        # Write to temporary file
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')

        # Write XML to models/stanford_tidybot/ directory (same directory as tidybot.xml)
        # This ensures ALL relative paths work exactly as they do in tidybot.xml
        project_root = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(project_root, 'models/stanford_tidybot')
        temp_file = os.path.join(temp_dir, f'.custom_scene_{os.getpid()}.xml')
        tree.write(temp_file, encoding='utf-8', xml_declaration=True)

        return temp_file

    def _add_object_assets(self, asset_element):
        """Add object mesh and texture assets to the scene."""
        # Get absolute path to project root
        project_root = os.path.dirname(os.path.abspath(__file__))

        for i, obj_spec in enumerate(self.objects_config):
            obj_name = self._get_object_name(obj_spec)

            # Check if it's a robocasa object (has model.xml)
            robocasa_path = os.path.join(project_root, f'object_assets/robocasa_objs/{obj_name}')
            objects_path = os.path.join(project_root, f'object_assets/objects/{obj_name}')

            if os.path.isdir(robocasa_path):
                # RoboCasa object - load visual mesh from model.xml
                import xml.etree.ElementTree as ET_parse
                model_xml_path = os.path.join(robocasa_path, 'model.xml')
                try:
                    tree = ET_parse.parse(model_xml_path)
                    root = tree.getroot()

                    # Find all mesh and texture assets
                    for mesh in root.findall('.//mesh'):
                        mesh_file = mesh.get('file')
                        if mesh_file:
                            # Convert to absolute path
                            abs_mesh_path = os.path.join(robocasa_path, mesh_file)
                            mesh_name = f'obj_{i}_{mesh.get("name", mesh_file.replace("/", "_"))}'
                            mesh.set('name', mesh_name)
                            mesh.set('file', abs_mesh_path)
                            asset_element.append(mesh)

                    # Find all texture assets
                    for texture in root.findall('.//texture'):
                        tex_file = texture.get('file')
                        if tex_file:
                            abs_tex_path = os.path.join(robocasa_path, tex_file)
                            tex_name = f'obj_{i}_{texture.get("name", tex_file.replace("/", "_"))}'
                            texture.set('name', tex_name)
                            texture.set('file', abs_tex_path)
                            asset_element.append(texture)

                    # Find materials
                    for material in root.findall('.//material'):
                        mat_name = f'obj_{i}_{material.get("name", "material")}'
                        material.set('name', mat_name)
                        # Update texture reference if exists
                        if material.get('texture'):
                            material.set('texture', f'obj_{i}_{material.get("texture")}')
                        asset_element.append(material)

                except Exception as e:
                    print(f"Warning: Could not load RoboCasa object {obj_name}: {e}")

            elif os.path.exists(objects_path):
                # Check file extension - MuJoCo only supports STL, OBJ, MSH
                if objects_path.endswith('.glb'):
                    print(f"ERROR: MuJoCo does not support GLB files: {obj_name}")
                    print(f"  GLB files need to be converted to STL or OBJ format.")
                    print(f"  Please use RoboCasa objects instead (e.g., 'apple_1' instead of 'apple.glb')")
                    continue
                elif objects_path.endswith(('.stl', '.obj', '.msh')):
                    # Supported format - add as mesh
                    mesh_name = f'obj_{i}_mesh'
                    scale = obj_spec.get('scale', 0.02) if isinstance(obj_spec, dict) else 0.02
                    ET.SubElement(asset_element, 'mesh',
                                 name=mesh_name,
                                 file=objects_path,
                                 scale=f'{scale} {scale} {scale}')
                else:
                    print(f"Warning: Unknown mesh format for {obj_name}")

    def _add_objects_to_worldbody(self, worldbody):
        """Add object bodies to the worldbody."""
        # Get absolute path to project root
        project_root = os.path.dirname(os.path.abspath(__file__))

        for i, obj_spec in enumerate(self.objects_config):
            obj_name = self._get_object_name(obj_spec)

            # Default position
            if isinstance(obj_spec, dict) and 'pos' in obj_spec:
                pos = obj_spec['pos']
            else:
                # Spread objects in a line
                pos = [0.5 + i * 0.1, -0.1 + i * 0.1, 0.02]

            pos_str = f'{pos[0]} {pos[1]} {pos[2]}'

            # Check object type (use absolute paths)
            robocasa_path = os.path.join(project_root, f'object_assets/robocasa_objs/{obj_name}')
            objects_path = os.path.join(project_root, f'object_assets/objects/{obj_name}')

            if os.path.isdir(robocasa_path):
                # RoboCasa object - load geometry from model.xml
                import xml.etree.ElementTree as ET_parse
                model_xml_path = os.path.join(robocasa_path, 'model.xml')
                try:
                    tree = ET_parse.parse(model_xml_path)
                    root = tree.getroot()

                    # Create body
                    body = ET.SubElement(worldbody, 'body', name=f'object_{i}', pos=pos_str)
                    ET.SubElement(body, 'freejoint')

                    # Find the visual mesh from the model.xml
                    model_body = root.find('.//body[@name="object"]')
                    if model_body is not None:
                        for geom in model_body.findall('geom'):
                            # Update mesh name to match what we loaded in assets
                            if geom.get('mesh'):
                                original_mesh = geom.get('mesh')
                                geom.set('mesh', f'obj_{i}_{original_mesh}')
                            # Update material reference
                            if geom.get('material'):
                                original_mat = geom.get('material')
                                geom.set('material', f'obj_{i}_{original_mat}')
                            body.append(geom)
                    else:
                        # Fallback: use a simple sphere
                        ET.SubElement(body, 'geom', type='sphere', size='0.03',
                                     rgba='0.8 0.2 0.2 1', mass='0.1')

                except Exception as e:
                    print(f"Warning: Could not load RoboCasa body for {obj_name}: {e}")
                    # Fallback
                    body = ET.SubElement(worldbody, 'body', name=f'object_{i}', pos=pos_str)
                    ET.SubElement(body, 'freejoint')
                    ET.SubElement(body, 'geom', type='sphere', size='0.03',
                                 rgba='0.8 0.2 0.2 1', mass='0.1')

            elif os.path.exists(objects_path):
                # Skip GLB files (already warned in _add_object_assets)
                if objects_path.endswith('.glb'):
                    continue

                # Supported mesh format - create body
                body = ET.SubElement(worldbody, 'body', name=f'object_{i}', pos=pos_str)
                ET.SubElement(body, 'freejoint')

                mesh_name = f'obj_{i}_mesh'
                ET.SubElement(body, 'geom', type='mesh', mesh=mesh_name,
                             rgba='0.8 0.8 0.8 1', mass='0.1')
            else:
                # Unknown object type - create simple cube
                body = ET.SubElement(worldbody, 'body', name=f'object_{i}', pos=pos_str)
                ET.SubElement(body, 'freejoint')
                ET.SubElement(body, 'geom', type='box', size='0.02 0.02 0.02',
                             rgba='0.5 0.5 0.5 1', mass='0.1')

    def physics_loop(self):
        """Override physics loop to use custom sim."""
        from threading import Thread

        # Create custom sim
        sim = CustomizableMujocoSim(self.mjcf_path, self.command_queue,
                                    self.shm_state, self.object_names,
                                    show_viewer=self.show_viewer)

        # Start render loop
        if self.render_images:
            Thread(target=self.render_loop, args=(sim.model, sim.data), daemon=True).start()

        # Launch sim
        sim.launch()

    def get_obs(self):
        """Get observations including all custom objects."""
        arm_quat = self.shm_state.arm_quat[[1, 2, 3, 0]]  # (w, x, y, z) -> (x, y, z, w)
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)

        obs = {
            'base_pose': self.shm_state.base_pose.copy(),
            'arm_pos': self.shm_state.arm_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.shm_state.gripper_pos.copy(),
        }
        obs['arm_pos'][0] -= 0.12  # Base link offset

        # Add all object positions and quaternions
        for i, obj_name in enumerate(self.object_names):
            obj_pos = self.shm_state.object_positions[i].copy()
            obj_quat = self.shm_state.object_quaternions[i][[1, 2, 3, 0]].copy()  # (w, x, y, z) -> (x, y, z, w)
            if obj_quat[3] < 0.0:  # Enforce quaternion uniqueness
                np.negative(obj_quat, out=obj_quat)

            obs[f'{obj_name}_pos'] = obj_pos
            obs[f'{obj_name}_quat'] = obj_quat

        # Add camera images
        if self.render_images:
            for shm_image in self.shm_images:
                obs[f'{shm_image.camera_name}_image'] = shm_image.data.copy()

        return obs


    def render_first_frame(self, output_dir='env_renders', prefix='env', save_state=True):
        """
        Render and save the first frame images from all cameras.

        Args:
            output_dir: Directory to save rendered images
            prefix: Prefix for saved image filenames
            save_state: Whether to save object state information

        Returns:
            Dictionary with camera names as keys and image arrays as values
        """
        import os
        import cv2 as cv
        import json

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Reset environment to get first frame
        print("Resetting environment for rendering...")
        self.reset()

        # Wait a bit for rendering to initialize
        import time
        time.sleep(0.5)

        # Get observations
        print("Capturing observations...")
        obs = self.get_obs()

        # Save camera images
        rendered_images = {}
        for key, value in obs.items():
            if key.endswith('_image'):
                camera_name = key.replace('_image', '')
                image_path = os.path.join(output_dir, f'{prefix}_{camera_name}.png')

                # Convert RGB to BGR for OpenCV
                image_bgr = cv.cvtColor(value, cv.COLOR_RGB2BGR)
                cv.imwrite(image_path, image_bgr)
                rendered_images[camera_name] = value
                print(f"  Saved {camera_name} image to {image_path}")

        # Save state information if requested
        if save_state:
            state_info = {
                'floor_texture': self.floor_texture,
                'num_objects': self.num_objects,
                'objects': [],
                'robot_state': {
                    'base_pose': obs['base_pose'].tolist(),
                    'arm_pos': obs['arm_pos'].tolist(),
                    'arm_quat': obs['arm_quat'].tolist(),
                    'gripper_pos': obs['gripper_pos'].tolist(),
                }
            }

            # Add object information
            for obj_name in self.object_names:
                state_info['objects'].append({
                    'name': obj_name,
                    'pos': obs[f'{obj_name}_pos'].tolist(),
                    'quat': obs[f'{obj_name}_quat'].tolist(),
                })

            state_path = os.path.join(output_dir, f'{prefix}_state.json')
            with open(state_path, 'w') as f:
                json.dump(state_info, f, indent=2)
            print(f"  Saved state information to {state_path}")

        print(f"Offline rendering complete! Saved to {output_dir}/")
        return rendered_images


if __name__ == '__main__':
    import time
    from constants import POLICY_CONTROL_PERIOD

    # Example 1: Custom floor texture with simple objects
    print("\n=== Example 1: Custom floor with GLB objects ===")
    env = CustomizableMujocoEnv(
        floor_texture='light_wood_v3.png',
        objects=['apple.glb', 'banana.glb', 'tomato.glb'],
        show_viewer=True,
        show_images=False
    )

    try:
        for episode in range(2):
            print(f"\nEpisode {episode + 1}")
            env.reset()
            for step in range(50):
                action = {
                    'base_pose': 0.1 * np.random.rand(3) - 0.05,
                    'arm_pos': 0.1 * np.random.rand(3) + np.array([0.55, 0.0, 0.4]),
                    'arm_quat': np.random.rand(4),
                    'gripper_pos': np.random.rand(1),
                }
                env.step(action)
                obs = env.get_obs()

                if step == 0:
                    print("Observation keys:", list(obs.keys()))
                    for key in obs.keys():
                        if not key.endswith('_image'):
                            print(f"  {key}: {obs[key]}")

                time.sleep(POLICY_CONTROL_PERIOD)
    finally:
        env.close()
