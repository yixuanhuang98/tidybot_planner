# Headless Mode Conversion Summary

This document summarizes the changes made to convert the TidyBot planner codebase to headless operation by disabling all GUI rendering components while enabling image saving to disk.

## Files Modified

### 1. `main.py`
**Changes**: 
- Always create headless MuJoCo environment when using simulation mode
- **Added**: `--save-images` argument to enable image saving in headless mode
- **Before**: Environment creation depended on `--teleop` flag
- **After**: Uses `MujocoEnv(render_images=render_images, show_viewer=False, show_images=False, save_images=args.save_images)`

### 2. `mujoco_env.py` 
**Changes**: 
- **MujocoEnv class defaults**: Changed from `(render_images=True, show_viewer=True, show_images=False)` to `(render_images=False, show_viewer=False, show_images=False)`
- **Added**: `save_images` parameter and automatic image saving functionality
- **Added**: Image saving in render loop with configurable intervals
- **Main section**: Now uses headless parameters with image saving by default

### 3. `replay_episodes.py`
**Changes**: 
- Use headless mode by default, only enable image rendering if `--show-images` or `--save-images` is explicitly requested
- **Added**: `--save-images` argument to save episode images to disk
- **Added**: Organized output directory structure for saved episode images
- **Before**: Always created environment with `render_images=False`
- **After**: Uses `render_images=args.show_images or args.save_images` and `show_viewer=False`

### 4. `cameras.py`
**Changes**: 
- **Main section**: Replaced OpenCV window display with automatic image saving
- **Added**: Timestamped directory structure for saved camera images
- **Added**: Configurable save intervals (default: every 2 seconds)
- **Added**: Status output showing saved image count and location

### 5. `plot_base_state.py`
**Changes**:
- **Added**: Non-interactive matplotlib backend (`matplotlib.use('Agg')`)
- **Modified Visualizer class**: Added `headless=True` parameter
- **Headless mode**: Prints robot position/velocity to console instead of plotting
- **Main loop**: Now runs in headless mode by default

### 6. `image_saver_config.py` (NEW)
**Features**:
- Configuration class for image saving settings
- Utility functions for organizing and managing saved images
- Automatic cleanup of old image sessions
- Image statistics and summary reporting
- Metadata saving capabilities

## GUI Components Disabled

1. **MuJoCo 3D Viewer**: The interactive 3D simulation window
2. **Camera Image Windows**: OpenCV windows showing camera feeds
3. **Image Visualization**: OpenCV windows during episode replay
4. **Matplotlib Plots**: Robot state visualization plots

## Image Saving Features Added

### 1. **Camera Image Saving**
- Automatically saves camera feeds to timestamped directories
- Configurable save intervals (default: every 2 seconds)
- Organized directory structure: `camera_images/YYYYMMDDTHHMMSS/`
- Status output showing save progress

**Usage:**
```bash
python cameras.py  # Saves to camera_images/ directory
```

### 2. **Simulation Image Saving**
- Saves rendered simulation camera views to disk
- Configurable save intervals (default: every 1 second) 
- Multiple camera angles saved simultaneously
- Directory structure: `simulation_images/YYYYMMDDTHHMMSS/`

**Usage:**
```bash
python main.py --sim --motion_planner --save-images
```

### 3. **Episode Replay Image Saving**
- Saves all images from recorded episodes to disk
- Organized by episode with step numbering
- Directory structure: `replay_images/YYYYMMDDTHHMMSS/replay_EPISODE/`

**Usage:**
```bash
python replay_episodes.py --sim --save-images --input-dir data/demos
```

### 4. **Image Management Utilities**
- Automatic cleanup of old image sessions
- Image statistics and disk usage reporting
- Configurable JPEG quality and file organization

**Usage:**
```bash
python image_saver_config.py  # View summary and cleanup old images
```

## Image Output Directory Structure

```
saved_images/
├── camera_images/
│   └── 20241201T143022/
│       ├── base_camera_20241201_143022_123_000001.jpg
│       └── wrist_camera_20241201_143022_123_000001.jpg
├── simulation_images/
│   └── 20241201T143022/
│       ├── base1_20241201_143022_456_000001.jpg
│       └── base2_20241201_143022_456_000001.jpg
└── replay_images/
    └── 20241201T143022/
        └── replay_20241201T120000000000/
            ├── base1_image_step_000001.jpg
            └── base2_image_step_000001.jpg
```

## How to Re-enable GUI Components

### MuJoCo Viewer
In `main.py` or when creating `MujocoEnv`, change:
```python
env = MujocoEnv(render_images=False, show_viewer=False, show_images=False)
```
to:
```python
env = MujocoEnv(render_images=True, show_viewer=True, show_images=True)
```

### Camera Windows
In `cameras.py`, replace the image saving loop with:
```python
cv.imshow('base_image', cv.cvtColor(base_image, cv.COLOR_RGB2BGR))
cv.imshow('wrist_image', cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
cv.waitKey(1)
```

### Episode Replay Visualization
Use the `--show-images` flag:
```bash
python replay_episodes.py --sim --show-images
```

### Robot State Plotting
In `plot_base_state.py`, change:
```python
visualizer = Visualizer(headless=True)
```
to:
```python
visualizer = Visualizer(headless=False)
```

## Benefits of Headless Mode with Image Saving

1. **No X11/Display Required**: Can run on servers without GUI
2. **Reduced Resource Usage**: Lower CPU/GPU usage without real-time rendering
3. **SSH-Friendly**: Works over SSH without X11 forwarding
4. **Docker/Container Compatible**: Easier to containerize
5. **Faster Execution**: No rendering overhead for real-time display
6. **Remote Server Deployment**: Can run on headless cloud instances
7. **Data Collection**: Systematic capture of visual data for analysis
8. **Offline Processing**: Images saved for later analysis or debugging

## Performance Impact

- **Simulation Speed**: Faster execution due to no real-time GUI rendering
- **Memory Usage**: Reduced memory usage (no GUI buffers)
- **CPU Usage**: Lower CPU usage from disabled real-time OpenCV/matplotlib operations
- **Disk Usage**: Images saved to disk (configurable quality and cleanup)
- **Network**: No bandwidth used for real-time visualization

## Image Saving Configuration

### Adjusting Save Intervals
Edit the constants in `image_saver_config.py`:
```python
CAMERA_SAVE_INTERVAL = 2.0  # seconds
SIMULATION_SAVE_INTERVAL = 1.0  # seconds
IMAGE_QUALITY = 95  # JPEG quality (0-100)
```

### Managing Disk Space
The system includes automatic cleanup:
- Keeps only the most recent sessions (configurable)
- JPEG compression with adjustable quality
- Maximum images per session limit
- Disk usage monitoring and reporting

## Testing the Image Saving

To test the image saving functionality:

```bash
# Test simulation with image saving
python main.py --sim --motion_planner --save-images

# Test episode replay with image saving
python replay_episodes.py --sim --save-images --input-dir data/demos

# Test camera system with image saving (if hardware available)
python cameras.py

# View image statistics and cleanup old files
python image_saver_config.py

# Test robot state visualization in headless mode
python plot_base_state.py
```

All commands will run without opening GUI windows while systematically saving visual data to organized directory structures.

## Troubleshooting

### Common Issues:
1. **Disk Space**: Monitor disk usage with `image_saver_config.py`
2. **Performance**: Adjust save intervals if system is too slow
3. **Image Quality**: Modify JPEG quality settings for size vs quality trade-off
4. **Directory Permissions**: Ensure write permissions in output directories

### Monitoring Resource Usage:
```bash
# Check disk usage
du -sh saved_images/

# Monitor system resources while running
htop

# Check image saving progress
tail -f <log_output>
``` 