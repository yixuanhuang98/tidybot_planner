#!/usr/bin/env python3
# Author: Assistant
# Date: 2024
#
# Test script for validating image saving functionality in headless mode

import argparse
import time
import subprocess
import sys
from pathlib import Path
from image_saver_config import print_image_summary, get_image_stats

def test_simulation_image_saving():
    """Test simulation image saving"""
    print("\n" + "="*60)
    print("TESTING SIMULATION IMAGE SAVING")
    print("="*60)
    
    print("Starting simulation with image saving for 10 seconds...")
    try:
        # Run simulation with image saving for 10 seconds
        proc = subprocess.Popen([
            sys.executable, "main.py", 
            "--sim", "--motion_planner", "--save-images"
        ])
        
        # Let it run for 10 seconds
        time.sleep(10)
        proc.terminate()
        proc.wait()
        
        # Check results
        sim_dir = Path("simulation_images")
        stats = get_image_stats(sim_dir)
        print(f"✅ Simulation test completed:")
        print(f"   Images saved: {stats['total_images']}")
        print(f"   Size: {stats['total_size_mb']:.1f} MB")
        print(f"   Location: {stats['directory']}")
        
        return stats['total_images'] > 0
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False

def test_camera_image_saving():
    """Test camera image saving (if hardware available)"""
    print("\n" + "="*60)
    print("TESTING CAMERA IMAGE SAVING")
    print("="*60)
    
    # Check if camera hardware might be available
    try:
        import cv2 as cv
        # Try to open a camera
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  No camera detected - skipping camera test")
            cap.release()
            return True
        cap.release()
    except:
        print("⚠️  Camera libraries not available - skipping camera test")
        return True
    
    print("Camera detected. Starting camera image saving for 10 seconds...")
    try:
        # Run camera saving for 10 seconds
        proc = subprocess.Popen([sys.executable, "cameras.py"])
        
        # Let it run for 10 seconds
        time.sleep(10)
        proc.terminate()
        proc.wait()
        
        # Check results
        cam_dir = Path("camera_images")
        stats = get_image_stats(cam_dir)
        print(f"✅ Camera test completed:")
        print(f"   Images saved: {stats['total_images']}")
        print(f"   Size: {stats['total_size_mb']:.1f} MB")
        print(f"   Location: {stats['directory']}")
        
        return stats['total_images'] > 0
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_episode_replay_saving():
    """Test episode replay image saving"""
    print("\n" + "="*60)
    print("TESTING EPISODE REPLAY IMAGE SAVING")
    print("="*60)
    
    # Check if demo episodes exist
    demo_dir = Path("data/demos")
    if not demo_dir.exists() or not any(demo_dir.iterdir()):
        print("⚠️  No demo episodes found - skipping replay test")
        return True
    
    print("Demo episodes found. Testing replay with image saving...")
    try:
        # Run episode replay with image saving
        proc = subprocess.Popen([
            sys.executable, "replay_episodes.py",
            "--sim", "--save-images", "--input-dir", "data/demos"
        ])
        
        # Let it run for 15 seconds
        time.sleep(15)
        proc.terminate()
        proc.wait()
        
        # Check results
        replay_dir = Path("replay_images")
        stats = get_image_stats(replay_dir)
        print(f"✅ Replay test completed:")
        print(f"   Images saved: {stats['total_images']}")
        print(f"   Size: {stats['total_size_mb']:.1f} MB")
        print(f"   Location: {stats['directory']}")
        
        return stats['total_images'] > 0
        
    except Exception as e:
        print(f"❌ Replay test failed: {e}")
        return False

def test_headless_operation():
    """Test that no GUI windows are opened"""
    print("\n" + "="*60)
    print("TESTING HEADLESS OPERATION")
    print("="*60)
    
    print("Checking that DISPLAY variable can be unset...")
    
    # Store original DISPLAY
    import os
    original_display = os.environ.get('DISPLAY')
    
    try:
        # Unset DISPLAY to ensure headless operation
        if 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
        
        # Try to run simulation without display
        proc = subprocess.Popen([
            sys.executable, "main.py",
            "--sim", "--motion_planner", "--save-images"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Let it run for 5 seconds
        time.sleep(5)
        proc.terminate()
        stdout, stderr = proc.communicate()
        
        # Check if it ran without display errors
        if b"DISPLAY" in stderr or b"display" in stderr:
            print("❌ Headless test failed: Display required")
            return False
        else:
            print("✅ Headless test passed: No display required")
            return True
            
    except Exception as e:
        print(f"❌ Headless test failed: {e}")
        return False
    finally:
        # Restore original DISPLAY
        if original_display:
            os.environ['DISPLAY'] = original_display

def main():
    parser = argparse.ArgumentParser(description="Test image saving functionality")
    parser.add_argument('--test-simulation', action='store_true', help='Test simulation image saving')
    parser.add_argument('--test-cameras', action='store_true', help='Test camera image saving')
    parser.add_argument('--test-replay', action='store_true', help='Test episode replay image saving')
    parser.add_argument('--test-headless', action='store_true', help='Test headless operation')
    parser.add_argument('--test-all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if not any([args.test_simulation, args.test_cameras, args.test_replay, args.test_headless, args.test_all]):
        args.test_all = True
    
    results = []
    
    print("🚀 STARTING IMAGE SAVING TESTS")
    print("This will test the headless image saving functionality")
    
    if args.test_all or args.test_simulation:
        results.append(("Simulation", test_simulation_image_saving()))
    
    if args.test_all or args.test_cameras:
        results.append(("Camera", test_camera_image_saving()))
    
    if args.test_all or args.test_replay:
        results.append(("Replay", test_episode_replay_saving()))
    
    if args.test_all or args.test_headless:
        results.append(("Headless", test_headless_operation()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:15} {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests PASSED! Image saving is working correctly.")
    else:
        print("\n⚠️  Some tests FAILED. Check the output above for details.")
    
    # Show overall image statistics
    print_image_summary()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main()) 