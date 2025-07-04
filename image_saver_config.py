# Author: Assistant
# Date: 2024
# 
# Image Saver Configuration and Utilities
# Provides configuration options and utilities for saving camera and simulation images

import os
import shutil
from pathlib import Path
from datetime import datetime
import cv2 as cv

class ImageSaverConfig:
    """Configuration class for image saving functionality"""
    
    # Default settings
    CAMERA_SAVE_INTERVAL = 2.0  # seconds
    SIMULATION_SAVE_INTERVAL = 1.0  # seconds
    IMAGE_QUALITY = 95  # JPEG quality (0-100)
    MAX_IMAGES_PER_SESSION = 1000  # Prevent disk space issues
    
    # Directory structure
    BASE_OUTPUT_DIR = Path('saved_images')
    CAMERA_SUBDIR = 'camera_images'
    SIMULATION_SUBDIR = 'simulation_images'
    REPLAY_SUBDIR = 'replay_images'

def create_session_directory(session_type='general'):
    """Create a timestamped directory for the current session"""
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    session_dir = ImageSaverConfig.BASE_OUTPUT_DIR / session_type / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir

def save_image_with_metadata(image, filepath, metadata=None):
    """Save image with optional metadata"""
    # Convert RGB to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume RGB input, convert to BGR for OpenCV
        bgr_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    else:
        bgr_image = image
    
    # Save with specified quality
    cv.imwrite(str(filepath), bgr_image, [cv.IMWRITE_JPEG_QUALITY, ImageSaverConfig.IMAGE_QUALITY])
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.with_suffix('.txt')
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

def cleanup_old_images(base_dir, keep_sessions=5):
    """Remove old image sessions, keeping only the most recent ones"""
    if not base_dir.exists():
        return
    
    # Get all session directories
    session_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove older sessions
    for old_dir in session_dirs[keep_sessions:]:
        print(f"Removing old image session: {old_dir}")
        shutil.rmtree(old_dir)

def get_image_stats(directory):
    """Get statistics about saved images in a directory"""
    if not directory.exists():
        return {"total_images": 0, "total_size_mb": 0}
    
    image_files = list(directory.rglob("*.jpg")) + list(directory.rglob("*.png"))
    total_size = sum(f.stat().st_size for f in image_files)
    
    return {
        "total_images": len(image_files),
        "total_size_mb": total_size / (1024 * 1024),
        "directory": str(directory)
    }

def print_image_summary():
    """Print a summary of all saved images"""
    print("\n" + "="*50)
    print("IMAGE SAVING SUMMARY")
    print("="*50)
    
    base_dir = ImageSaverConfig.BASE_OUTPUT_DIR
    
    for subdir_name in [ImageSaverConfig.CAMERA_SUBDIR, 
                        ImageSaverConfig.SIMULATION_SUBDIR, 
                        ImageSaverConfig.REPLAY_SUBDIR]:
        subdir = base_dir / subdir_name
        stats = get_image_stats(subdir)
        
        print(f"\n{subdir_name.upper()}:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total size: {stats['total_size_mb']:.1f} MB")
        print(f"  Location: {stats['directory']}")
    
    total_stats = get_image_stats(base_dir)
    print(f"\nTOTAL ACROSS ALL CATEGORIES:")
    print(f"  Total images: {total_stats['total_images']}")
    print(f"  Total size: {total_stats['total_size_mb']:.1f} MB")
    print("="*50)

if __name__ == "__main__":
    print("Image Saver Configuration Utility")
    print_image_summary()
    
    # Cleanup old sessions (keep last 3)
    for subdir_name in [ImageSaverConfig.CAMERA_SUBDIR, 
                        ImageSaverConfig.SIMULATION_SUBDIR, 
                        ImageSaverConfig.REPLAY_SUBDIR]:
        subdir = ImageSaverConfig.BASE_OUTPUT_DIR / subdir_name
        cleanup_old_images(subdir, keep_sessions=3) 