#!/usr/bin/env python3
"""
Demo script for MuJoCo WebRenderer
Shows how to integrate the web renderer with a MuJoCo environment
"""
import os
import sys
import time
import threading
import mujoco
import numpy as np

# Add the current directory to sys.path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_renderer import create_web_renderer


def create_simple_scene():
    """Create a simple MuJoCo scene for demonstration"""
    xml_content = """
    <mujoco>
        <compiler coordinate="local" inertiafromgeom="true"/>
        <option timestep="0.01"/>
        
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
            
            <body name="cube1" pos="0.5 0 0.5">
                <geom name="cube1_geom" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
                <joint name="cube1_joint" type="free"/>
            </body>
            
            <body name="cube2" pos="-0.5 0 0.5">
                <geom name="cube2_geom" type="box" size="0.1 0.1 0.1" rgba="0 1 0 1"/>
                <joint name="cube2_joint" type="free"/>
            </body>
            
            <body name="sphere" pos="0 0.5 0.5">
                <geom name="sphere_geom" type="sphere" size="0.1" rgba="0 0 1 1"/>
                <joint name="sphere_joint" type="free"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    # Create model from XML string
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    
    return model, data


def animate_scene(model, data, web_renderer):
    """Animate the scene by updating object positions"""
    print("Starting scene animation...")
    
    # Initial positions
    cube1_pos = np.array([0.5, 0, 0.5])
    cube2_pos = np.array([-0.5, 0, 0.5])
    sphere_pos = np.array([0, 0.5, 0.5])
    
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time() - start_time
            
            # Animate cube1 (circular motion)
            cube1_pos[0] = 0.5 * np.cos(current_time)
            cube1_pos[1] = 0.5 * np.sin(current_time)
            
            # Animate cube2 (up-down motion)
            cube2_pos[2] = 0.5 + 0.3 * np.sin(current_time * 2)
            
            # Animate sphere (figure-8 motion)
            sphere_pos[0] = 0.3 * np.sin(current_time * 1.5)
            sphere_pos[1] = 0.3 * np.sin(current_time * 3)
            
            # Update positions in the simulation
            if hasattr(data, 'qpos'):
                # Find joint indices
                cube1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube1_joint")
                cube2_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube2_joint")
                sphere_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "sphere_joint")
                
                # Update positions (free joints have 7 DOF: 3 translation + 4 quaternion)
                if cube1_joint_id >= 0:
                    qpos_start = model.jnt_qposadr[cube1_joint_id]
                    data.qpos[qpos_start:qpos_start+3] = cube1_pos
                    data.qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]  # Identity quaternion
                
                if cube2_joint_id >= 0:
                    qpos_start = model.jnt_qposadr[cube2_joint_id]
                    data.qpos[qpos_start:qpos_start+3] = cube2_pos
                    data.qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]  # Identity quaternion
                
                if sphere_joint_id >= 0:
                    qpos_start = model.jnt_qposadr[sphere_joint_id]
                    data.qpos[qpos_start:qpos_start+3] = sphere_pos
                    data.qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]  # Identity quaternion
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Control animation speed
            time.sleep(0.02)  # 50 FPS
            
    except KeyboardInterrupt:
        print("Animation stopped by user")
    except Exception as e:
        print(f"Animation error: {e}")


def main():
    """Main function to run the demo"""
    print("🤖 MuJoCo WebRenderer Demo")
    print("=" * 50)
    
    try:
        # Create a simple scene
        print("Creating MuJoCo scene...")
        model, data = create_simple_scene()
        print(f"✓ Scene created with {model.nbody} bodies")
        
        # Create web renderer
        print("Initializing WebRenderer...")
        web_renderer = create_web_renderer(model, data, width=640, height=480, port=5000)
        print("✓ WebRenderer initialized")
        
        # Start animation in a separate thread
        animation_thread = threading.Thread(target=animate_scene, args=(model, data, web_renderer), daemon=True)
        animation_thread.start()
        print("✓ Animation started")
        
        # Start the web server
        print("\n" + "=" * 50)
        print("🌐 Starting web server...")
        print("   URL: http://localhost:5000")
        print("   Press Ctrl+C to stop")
        print("=" * 50)
        
        # Run the web server (this will block until interrupted)
        web_renderer.run(host='0.0.0.0', debug=False)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Shutting down...")
        if 'web_renderer' in locals():
            web_renderer.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main() 