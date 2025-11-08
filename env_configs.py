"""
Environment configuration presets for customizable MuJoCo environments.

This module provides predefined environment configurations for common scenarios
such as pick-and-place tasks, sorting tasks, and data collection.
"""

import random
from typing import Dict, List, Optional, Union

# Floor texture presets
FLOOR_TEXTURES = {
    'wood_light': 'light_wood_v3.png',
    'wood_dark': 'wood_light.png',
    'metal': 'metal.png',
    'yellow_plaster': 'yellow-plaster.png',
    'pink_plaster': 'pink-plaster.png',
    'white_bricks': 'white-bricks.png',
    'stone': 'StoneWall13.png',
    'camouflage': 'Woodland_Camouflage.png',
}

# Object categories
FRUIT_OBJECTS = [
    'apple.glb', 'banana.glb', 'orange.glb', 'pear.glb',
    'tomato.glb', 'grape.glb', 'watermelon.glb'
]

VEGETABLE_OBJECTS = [
    'carrot.glb', 'cucumber.glb', 'eggplant.glb', 'broccoli.glb',
    'lettuce.glb', 'mushroom.glb', 'potato.glb', 'pumpkin.glb'
]

CONTAINER_OBJECTS = [
    'bowl.glb', 'plate.glb', 'cup.glb', 'bottle.glb'
]

TOOL_OBJECTS = [
    'knife.glb', 'spoon.glb', 'hammer.glb'
]

ROBOCASA_FRUITS = [
    'apple_1', 'apple_2', 'banana_1', 'banana_2',
    'orange_1', 'orange_2', 'pear_1', 'pear_2'
]

EASYGRASP_OBJECTS = [
    'bar_soap_0', 'boxed_drink_0', "cake_0"
]

# EASYGRASP_OBJECTS = [
#     'apple_0', 'bar_soap_0', 'boxed_drink_0', "bell_pepper_0", "cake_0",
#     "lemon_0", "orange_1", "potato_0", "water_bottle_0", "avocado_0"
# ]

ROBOCASA_VEGETABLES = [
    'carrot_0', 'carrot_1', 'cucumber_0', 'cucumber_1',
    'eggplant_0', 'eggplant_1', 'broccoli_0', 'broccoli_3'
]

ROBOCASA_CONTAINERS = [
    'bowl_0', 'bowl_2', 'bowl_3', 'plate_1', 'plate_3',
    'cup_2', 'cup_4', 'cup_5'
]


class EnvConfigBuilder:
    """Builder class for creating environment configurations."""

    def __init__(self):
        self.config = {
            'floor_texture': None,
            'objects': [],
            'render_images': True,
            'show_viewer': True,
            'show_images': False
        }

    def with_floor(self, texture: str) -> 'EnvConfigBuilder':
        """Set floor texture."""
        if texture in FLOOR_TEXTURES:
            self.config['floor_texture'] = FLOOR_TEXTURES[texture]
        else:
            self.config['floor_texture'] = texture
        return self

    def with_objects(self, objects: List[str]) -> 'EnvConfigBuilder':
        """Set specific objects."""
        self.config['objects'] = objects
        return self

    def with_random_objects(self, category: str, count: int = 3) -> 'EnvConfigBuilder':
        """Add random objects from a category."""
        categories = {
            'fruits': FRUIT_OBJECTS,
            'vegetables': VEGETABLE_OBJECTS,
            'containers': CONTAINER_OBJECTS,
            'tools': TOOL_OBJECTS,
            'robocasa_fruits': ROBOCASA_FRUITS,
            'robocasa_vegetables': ROBOCASA_VEGETABLES,
            'robocasa_containers': ROBOCASA_CONTAINERS,
            "easygrasp_objects": EASYGRASP_OBJECTS
        }

        if category not in categories:
            raise ValueError(f"Unknown category: {category}. Available: {list(categories.keys())}")

        available_objects = categories[category]
        selected_objects = random.sample(available_objects, min(count, len(available_objects)))
        self.config['objects'] = selected_objects
        return self

    def headless(self) -> 'EnvConfigBuilder':
        """Configure for headless operation (no viewer)."""
        self.config['show_viewer'] = True
        self.config['show_images'] = False
        return self

    def with_rendering(self, render: bool = True, show_images: bool = False) -> 'EnvConfigBuilder':
        """Configure rendering options."""
        self.config['render_images'] = render
        self.config['show_images'] = show_images
        return self

    def build(self) -> Dict:
        """Build and return the configuration dictionary."""
        return self.config.copy()


# Predefined environment configurations
ENV_CONFIGS = {
    # Basic pick-and-place with fruits
    'pick_place_fruits': {
        'floor_texture': FLOOR_TEXTURES['wood_light'],
        'objects': ['apple.glb', 'banana.glb', 'orange.glb'],
        'render_images': True,
        'show_viewer': True,
        'show_images': False
    },

    # Sorting task with mixed objects
    'sorting_mixed': {
        'floor_texture': FLOOR_TEXTURES['metal'],
        'objects': ['apple.glb', 'carrot.glb', 'bowl.glb', 'tomato.glb', 'cucumber.glb'],
        'render_images': True,
        'show_viewer': True,
        'show_images': False
    },

    # Dense clutter with many objects
    'dense_clutter': {
        'floor_texture': FLOOR_TEXTURES['yellow_plaster'],
        'objects': [
            'apple.glb', 'banana.glb', 'tomato.glb', 'carrot.glb',
            'cucumber.glb', 'bowl.glb', 'plate.glb', 'cup.glb'
        ],
        'render_images': True,
        'show_viewer': True,
        'show_images': False
    },

    # Minimal setup for quick testing
    'minimal': {
        'floor_texture': FLOOR_TEXTURES['wood_light'],
        'objects': ['apple.glb'],
        'render_images': False,
        'show_viewer': True,
        'show_images': False
    },

    # Headless for data collection
    'data_collection': {
        'floor_texture': FLOOR_TEXTURES['wood_light'],
        'objects': ['apple.glb', 'banana.glb', 'tomato.glb'],
        'render_images': True,
        'show_viewer': True,
        'show_images': False
    },

    # RoboCasa objects
    'robocasa_fruits': {
        'floor_texture': FLOOR_TEXTURES['wood_dark'],
        'objects': ['apple_1', 'banana_1', 'orange_1'],
        'render_images': True,
        'show_viewer': True,
        'show_images': False
    },

    # Kitchen scene with containers
    'kitchen_scene': {
        'floor_texture': FLOOR_TEXTURES['white_bricks'],
        'objects': [
            'bowl_0', 'plate_1', 'cup_2',
            'apple_1', 'banana_1', 'carrot_0'
        ],
        'render_images': True,
        'show_viewer': True,
        'show_images': False
    },

}


def get_config(name: str) -> Dict:
    """Get a predefined environment configuration by name."""
    if name not in ENV_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(ENV_CONFIGS.keys())}")
    return ENV_CONFIGS[name].copy()


def create_random_config(
    floor: Optional[str] = None,
    object_category: str = 'fruits',
    num_objects: int = 3,
    headless: bool = False
) -> Dict:
    """
    Create a random environment configuration.

    Args:
        floor: Floor texture name (from FLOOR_TEXTURES) or None for random
        object_category: Category to sample objects from
        num_objects: Number of objects to include
        headless: Whether to run without viewer

    Returns:
        Configuration dictionary
    """
    builder = EnvConfigBuilder()

    # Random floor if not specified
    if floor is None:
        floor = random.choice(list(FLOOR_TEXTURES.keys()))

    builder.with_floor(floor)
    builder.with_random_objects(object_category, num_objects)

    if headless:
        builder.headless()

    return builder.build()


def list_available_configs():
    """Print all available predefined configurations."""
    print("Available Environment Configurations:")
    print("=" * 60)
    for name, config in ENV_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Floor: {config['floor_texture']}")
        print(f"  Objects: {config['objects']}")
        print(f"  Render: {config['render_images']}, Viewer: {config['show_viewer']}")
    print("\n" + "=" * 60)


# Example usage
if __name__ == '__main__':
    from customizable_env import CustomizableMujocoEnv

    # List available configs
    list_available_configs()

    # Example 1: Use a predefined config
    print("\n\nExample 1: Using predefined 'pick_place_fruits' config")
    config = get_config('pick_place_fruits')
    print(f"Config: {config}")

    # Example 2: Build a custom config
    print("\n\nExample 2: Building custom config")
    config = (EnvConfigBuilder()
              .with_floor('metal')
              .with_objects(['apple.glb', 'bowl.glb'])
              .headless()
              .build())
    print(f"Config: {config}")

    # Example 3: Random config
    print("\n\nExample 3: Random configuration")
    config = create_random_config(
        object_category='vegetables',
        num_objects=4,
        headless=True
    )
    print(f"Config: {config}")

    # Example 4: Create environment with config
    print("\n\nExample 4: Creating environment (uncomment to run)")
    # env = CustomizableMujocoEnv(**get_config('minimal'))
    # env.reset()
    # obs = env.get_obs()
    # print(f"Observation keys: {list(obs.keys())}")
    # env.close()
