#!/usr/bin/env python3
"""
Script to load the cabinet environment and output the poses of the door handles.
"""

import mujoco
import numpy as np
import os

def load_cabinet_scene():
    """Load the cabinet scene and return the model and data."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "cabinet.xml")
    
    # Load the model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    return model, data

def get_handle_poses(model, data):
    """Extract the poses of the door handles."""
    # Forward kinematics to update all positions
    mujoco.mj_forward(model, data)
    
    # Get site IDs
    try:
        leftdoor_site_id = model.site("leftdoor_site").id
        rightdoor_site_id = model.site("rightdoor_site").id
    except:
        # Fallback: find sites by name
        leftdoor_site_id = None
        rightdoor_site_id = None
        for i in range(model.nsite):
            site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if site_name == "leftdoor_site":
                leftdoor_site_id = i
            elif site_name == "rightdoor_site":
                rightdoor_site_id = i
    
    if leftdoor_site_id is None or rightdoor_site_id is None:
        print("Error: Could not find door handle sites")
        return None, None
    
    # Get world positions of the sites
    leftdoor_pos = data.site_xpos[leftdoor_site_id].copy()
    rightdoor_pos = data.site_xpos[rightdoor_site_id].copy()
    
    # Get world orientations of the sites (as rotation matrices)
    leftdoor_rot = data.site_xmat[leftdoor_site_id].reshape(3, 3).copy()
    rightdoor_rot = data.site_xmat[rightdoor_site_id].reshape(3, 3).copy()
    
    return (leftdoor_pos, leftdoor_rot), (rightdoor_pos, rightdoor_rot)

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (ZYX convention)."""
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    
    return np.array([x, y, z])

def main():
    """Main function to load the scene and print handle poses."""
    print("Loading cabinet environment...")
    
    try:
        model, data = load_cabinet_scene()
        print(f"✓ Successfully loaded cabinet environment")
        print(f"  - Model has {model.nbody} bodies")
        print(f"  - Model has {model.nsite} sites")
        print(f"  - Model has {model.njnt} joints")
        
        # Get handle poses
        left_handle, right_handle = get_handle_poses(model, data)
        
        if left_handle is None or right_handle is None:
            print("✗ Failed to get handle poses")
            return
        
        left_pos, left_rot = left_handle
        right_pos, right_rot = right_handle
        
        # Convert rotation matrices to Euler angles
        left_euler = rotation_matrix_to_euler(left_rot)
        right_euler = rotation_matrix_to_euler(right_rot)
        
        print("\n" + "="*60)
        print("DOOR HANDLE POSES")
        print("="*60)
        
        print(f"\nLEFT DOOR HANDLE:")
        print(f"  Position (x, y, z): [{left_pos[0]:.6f}, {left_pos[1]:.6f}, {left_pos[2]:.6f}]")
        print(f"  Euler angles (rad): [{left_euler[0]:.6f}, {left_euler[1]:.6f}, {left_euler[2]:.6f}]")
        print(f"  Euler angles (deg): [{np.degrees(left_euler[0]):.2f}°, {np.degrees(left_euler[1]):.2f}°, {np.degrees(left_euler[2]):.2f}°]")
        
        print(f"\nRIGHT DOOR HANDLE:")
        print(f"  Position (x, y, z): [{right_pos[0]:.6f}, {right_pos[1]:.6f}, {right_pos[2]:.6f}]")
        print(f"  Euler angles (rad): [{right_euler[0]:.6f}, {right_euler[1]:.6f}, {right_euler[2]:.6f}]")
        print(f"  Euler angles (deg): [{np.degrees(right_euler[0]):.2f}°, {np.degrees(right_euler[1]):.2f}°, {np.degrees(right_euler[2]):.2f}°]")
        
        print("\n" + "="*60)
        
        # Additional info about cabinet pose
        interactive_obj_id = None
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name == "interactive_obj":
                interactive_obj_id = i
                break
        
        if interactive_obj_id is not None:
            cabinet_pos = data.xpos[interactive_obj_id]
            cabinet_quat = data.xquat[interactive_obj_id]
            print(f"CABINET BASE POSE:")
            print(f"  Position: [{cabinet_pos[0]:.6f}, {cabinet_pos[1]:.6f}, {cabinet_pos[2]:.6f}]")
            print(f"  Quaternion (w,x,y,z): [{cabinet_quat[0]:.6f}, {cabinet_quat[1]:.6f}, {cabinet_quat[2]:.6f}, {cabinet_quat[3]:.6f}]")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 