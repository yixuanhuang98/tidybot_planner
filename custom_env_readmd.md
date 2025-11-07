# Customizable Environment Guide

This guide shows how to use the customizable environment system with high-quality RoboCasa objects and custom textures.

## Asset Locations

- **Floor textures**: `object_assets/use_textures/`
- **Objects**: `object_assets/robocasa_objs/`

## Quick Start

### Basic Command Line Usage

```bash
# Standard simulation with teleop
python main.py --sim --teleop

# Custom environment
python main.py --sim --teleop --custom-env --random-env \
    --object-category easygrasp_objects \
    --num-objects 4

# Motion planner mode
python main.py --sim --motion_planner --custom-env --random-env \
    --object-category easygrasp_objects \
    --num-objects 4
```

## Command Line Arguments

### Custom Floor Textures and Objects

Specify floor texture with `--floor-texture`, specify objects with `--objects` (comma-separated list):

```bash
# Light wood floor
python main.py --sim --custom-env --floor-texture light_wood_v3.png \
    --objects apple_0,avocado_0

# Metal floor
python main.py --sim --custom-env --floor-texture metal.png \
    --objects apple_0,avocado_0

# Yellow plaster floor
python main.py --sim --custom-env --floor-texture yellow-plaster.png \
    --objects apple_0,avocado_0
```

**Available textures** (in `object_assets/use_textures/`):
- `light_wood_v3.png`, `wood_light.png`
- `metal.png`
- `yellow-plaster.png`, `pink-plaster.png`
- `white-bricks.png`
- `StoneWall13.png`

### Random Environment Generation

Use `--random-env` with `--object-category` and `--num-objects`:

```bash
# Random RoboCasa graspable objects
python main.py --sim --teleop --custom-env --random-env \
    --object-category easygrasp_objects \
    --num-objects 4
```

### Using Predefined Configurations

The `--env-config` argument loads predefined environment setups. You can create new configs in `env_configs.py`.

<!-- **Available predefined configs right now:**
- `robocasa_fruits` - RoboCasa fruit objects
- `kitchen_scene` - Kitchen containers and food items
- `sorting_mixed` - Mixed objects on metal floor
- `dense_clutter` - 8 objects on yellow plaster
- `data_collection` - Headless mode for data collection -->

### Python API with Configurations

```python
from customizable_env import CustomizableMujocoEnv
from env_configs import get_config

# Load predefined config
config = get_config('robocasa_fruits')
env = CustomizableMujocoEnv(**config)

# Use environment
env.reset()
obs = env.get_obs()
action = {
    'base_pose': [0.0, 0.0, 0.0],
    'arm_pos': [0.55, 0.0, 0.4],
    'arm_quat': [0, 0, 0, 1],
    'gripper_pos': [0.5],
}
env.step(action)
env.close()
```

### Creating Custom Configurations

Add your own configurations to `env_configs.py`:

```python
ENV_CONFIGS['my_custom_task'] = {
    'floor_texture': 'metal.png',
    'objects': ['apple_1', 'banana_1', 'bowl_0', 'plate_1'],
    'render_images': True,
    'show_viewer': True,
    'show_images': False
}
```

Then use it:

```bash
python main.py --sim --teleop --custom-env --env-config my_custom_task
```

### Random Configuration Generation

```python
from customizable_env import CustomizableMujocoEnv
from env_configs import create_random_config

# Create random RoboCasa vegetables environment
config = create_random_config(
    object_category='robocasa_vegetables',
    num_objects=4,
    headless=True
)
env = CustomizableMujocoEnv(**config)
```

### List Available Configurations

```python
from env_configs import list_available_configs

# Print all available configs
list_available_configs()
```

Or via command line:

```bash
python env_configs.py
```

## Data Collection

### Basic Data Collection

```bash
# Collect data with custom environment
python main.py --sim --teleop --custom-env \
    --env-config robocasa_fruits \
    --save --output-dir data/custom_demos

# Headless data collection (faster)
python main.py --sim --motion_planner --custom-env \
    --env-config data_collection \
    --save --output-dir data/auto_demos
```

### Domain Randomization for Training

Generate diverse training data with random environments:

```bash
# Collect 100 demonstrations with random RoboCasa fruits
for i in {1..100}; do
    python main.py --sim --motion_planner --custom-env --random-env \
        --object-category robocasa_fruits \
        --num-objects 3 \
        --save --output-dir data/diverse_demos
done
```

### Batch Collection Script

```bash
#!/bin/bash
# Collect data with multiple RoboCasa object categories

CATEGORIES=("robocasa_fruits" "robocasa_vegetables" "robocasa_containers")

for category in "${CATEGORIES[@]}"; do
    echo "Collecting data for category: $category"
    for i in {1..50}; do
        python main.py --sim --motion_planner --custom-env --random-env \
            --object-category $category \
            --num-objects 4 \
            --save --output-dir data/multi_category_demos/$category
    done
done
```

## Offline Rendering

### Single Scene Rendering

Render first frame without running full simulation using `--render-only`:

```bash
# Render predefined config
python main.py --sim --custom-env --env-config robocasa_fruits --render-only

# Render custom RoboCasa scene
python main.py --sim --custom-env \
    --floor-texture metal.png \
    --objects apple_1,banana_1,bowl_0 \
    --render-only

# Render with custom output location
python main.py --sim --custom-env \
    --floor-texture light_wood_v3.png \
    --objects apple_1,banana_1,carrot_3,bowl_0 \
    --render-only \
    --render-output-dir my_renders \
    --render-prefix kitchen_scene

# Render random RoboCasa configuration
python main.py --sim --custom-env --random-env \
    --object-category robocasa_vegetables \
    --num-objects 4 \
    --render-only
```

**Output files:**
- `env_renders/{prefix}_{camera_name}.png` - Camera images
- `env_renders/{prefix}_state.json` - Object state and configuration

### Batch Rendering

Use `batch_render_envs.py` to render multiple configurations:

```bash
# Render all predefined configs
python batch_render_envs.py --mode predefined

# Render all floor textures (with default RoboCasa objects)
python batch_render_envs.py --mode floors

# Render all RoboCasa object categories
python batch_render_envs.py --mode objects

# Render N random variations
python batch_render_envs.py --mode random --num-random 10

# Render everything
python batch_render_envs.py --mode all --num-random 5

# Create HTML gallery
python batch_render_envs.py --mode all --create-gallery

# Custom batch rendering
python batch_render_envs.py --mode custom \
    --floor-texture yellow-plaster.png \
    --objects apple_1,carrot_3,bowl_0 \
    --prefix custom_scene \
    --output-dir custom_renders
```

**View gallery:**

```bash
firefox env_renders/gallery.html
```

### Python API for Rendering

```python
from customizable_env import CustomizableMujocoEnv

# Create environment for rendering
env = CustomizableMujocoEnv(
    floor_texture='metal.png',
    objects=['apple_1', 'banana_1', 'bowl_0'],
    render_images=True,
    show_viewer=False
)

# Render first frame
images = env.render_first_frame(
    output_dir='renders',
    prefix='robocasa_scene_01',
    save_state=True
)

env.close()
```

## Advanced Usage

### Curriculum Learning Setup

```bash
# Stage 1: Single RoboCasa object
python main.py --sim --motion_planner --custom-env \
    --objects apple_1 \
    --save --output-dir data/curriculum/stage1

# Stage 2: Two RoboCasa objects
python main.py --sim --motion_planner --custom-env \
    --objects apple_1,banana_1 \
    --save --output-dir data/curriculum/stage2

# Stage 3: Multiple objects with container
python main.py --sim --motion_planner --custom-env \
    --objects apple_1,banana_1,carrot_3,bowl_0 \
    --save --output-dir data/curriculum/stage3

# Stage 4: Complex scene
python main.py --sim --motion_planner --custom-env \
    --env-config dense_clutter \
    --save --output-dir data/curriculum/stage4
```

### Multi-Environment Training

```bash
#!/bin/bash
# Collect data across different environment configurations

CONFIGS=("robocasa_fruits" "kitchen_scene")

for config in "${CONFIGS[@]}"; do
    echo "Collecting data for config: $config"
    python main.py --sim --motion_planner --custom-env \
        --env-config $config \
        --save --output-dir data/multi_env_demos/$config
done
```

### Progressive Complexity

```bash
#!/bin/bash
# Gradually increase environment complexity

# Simple: 2 objects
python main.py --sim --motion_planner --custom-env --random-env \
    --object-category robocasa_fruits --num-objects 2 \
    --save --output-dir data/progressive/simple

# Medium: 4 objects
python main.py --sim --motion_planner --custom-env --random-env \
    --object-category robocasa_fruits --num-objects 4 \
    --save --output-dir data/progressive/medium

# Complex: 6 objects
python main.py --sim --motion_planner --custom-env --random-env \
    --object-category robocasa_fruits --num-objects 6 \
    --save --output-dir data/progressive/complex
```

### Testing and Debugging

```bash
# Test offline rendering
python test_offline_render.py

# List available configurations
python env_configs.py

# Render for visual verification
python main.py --sim --custom-env \
    --objects apple_1,banana_1,bowl_0 \
    --render-only

# Check rendered images
ls -lh env_renders/
```

## Common Workflows

### 1. Test New Environment Configuration

```bash
# Create and render to verify setup
python main.py --sim --custom-env \
    --floor-texture metal.png \
    --objects apple_1,carrot_3,bowl_0 \
    --render-only

# Check rendered images in env_renders/

# If looks good, run interactively
python main.py --sim --teleop --custom-env \
    --floor-texture metal.png \
    --objects apple_1,carrot_3,bowl_0
```

### 2. Generate Dataset Visualization

```bash
# Render multiple random RoboCasa variations
python batch_render_envs.py --mode random --num-random 20 --create-gallery

# View gallery
firefox env_renders/gallery.html
```

### 3. Create Documentation Images

```bash
# Render specific RoboCasa scenes
python main.py --sim --custom-env \
    --env-config robocasa_fruits \
    --render-only \
    --render-prefix docs_robocasa_fruits

python main.py --sim --custom-env \
    --env-config kitchen_scene \
    --render-only \
    --render-prefix docs_kitchen_scene
```

## Performance Optimization

### Faster Execution

```bash
# Disable rendering for faster execution
python main.py --sim --motion_planner --custom-env \
    --env-config data_collection \
    --save --output-dir data/fast_demos

# Use fewer objects
python main.py --sim --teleop --custom-env \
    --objects apple_1,banana_1
```

### Headless Mode

```bash
# Use data_collection config for headless operation
python main.py --sim --motion_planner --custom-env \
    --env-config data_collection \
    --save --output-dir data/headless_demos
```

## Troubleshooting

**Issue: Objects not appearing**
- Check that object files exist in `object_assets/robocasa_objs/`
- Verify object names match available RoboCasa objects

**Issue: Texture not loading**
- Verify texture file exists in `object_assets/use_textures/`
- Use one of the available textures listed above

**Issue: Slow performance**
- Use fewer objects with `--num-objects`
- Enable headless mode with `--env-config data_collection`

**Issue: ModuleNotFoundError**
- Install required dependencies: `pip install mujoco`

## Getting Help

```bash
# Main script help
python main.py --help

# Batch renderer help
python batch_render_envs.py --help

# List available configs
python env_configs.py

# Run tests
python test_customizable_env.py
```

## File Locations

```
tidybot_planner/
├── customizable_env.py          # Main environment class
├── env_configs.py                # Configuration presets
├── batch_render_envs.py         # Batch rendering utility
├── test_customizable_env.py     # Test suite
├── main.py                       # Main entry point
├── object_assets/
│   ├── use_textures/            # Floor textures (PNG)
│   └── robocasa_objs/           # RoboCasa objects
└── env_renders/                 # Default render output directory
```
