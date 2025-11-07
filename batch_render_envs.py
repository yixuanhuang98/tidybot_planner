#!/usr/bin/env python3
"""
Batch render multiple environment configurations.

This script renders the first frame of various environment configurations
for visualization and documentation purposes.
"""

import argparse
import os
from customizable_env import CustomizableMujocoEnv
from env_configs import (
    ENV_CONFIGS, FLOOR_TEXTURES, FRUIT_OBJECTS, VEGETABLE_OBJECTS,
    CONTAINER_OBJECTS, create_random_config, EnvConfigBuilder
)


def render_all_predefined_configs(output_dir='env_renders/predefined'):
    """Render all predefined environment configurations."""
    print("\n=== Rendering All Predefined Configurations ===\n")

    for config_name in ENV_CONFIGS.keys():
        print(f"Rendering config: {config_name}")
        try:
            config = ENV_CONFIGS[config_name].copy()
            # Force rendering and disable viewer for batch mode
            config['render_images'] = True
            config['show_viewer'] = False
            config['show_images'] = False

            env = CustomizableMujocoEnv(**config)
            env.render_first_frame(
                output_dir=output_dir,
                prefix=config_name,
                save_state=True
            )
            env.close()
            print(f"  ✓ Successfully rendered {config_name}\n")
        except Exception as e:
            print(f"  ✗ Failed to render {config_name}: {e}\n")


def render_all_floor_textures(output_dir='env_renders/floors'):
    """Render the same scene with all available floor textures."""
    print("\n=== Rendering All Floor Textures ===\n")

    # Use simple object set for consistency
    objects = ['apple.glb', 'banana.glb', 'tomato.glb']

    for texture_name, texture_file in FLOOR_TEXTURES.items():
        print(f"Rendering floor texture: {texture_name}")
        try:
            env = CustomizableMujocoEnv(
                floor_texture=texture_file,
                objects=objects,
                render_images=True,
                show_viewer=False,
                show_images=False
            )
            env.render_first_frame(
                output_dir=output_dir,
                prefix=f'floor_{texture_name}',
                save_state=True
            )
            env.close()
            print(f"  ✓ Successfully rendered {texture_name}\n")
        except Exception as e:
            print(f"  ✗ Failed to render {texture_name}: {e}\n")


def render_object_categories(output_dir='env_renders/objects'):
    """Render different object categories."""
    print("\n=== Rendering Object Categories ===\n")

    categories = {
        'fruits': FRUIT_OBJECTS[:3],
        'vegetables': VEGETABLE_OBJECTS[:3],
        'containers': CONTAINER_OBJECTS[:3],
        'mixed': ['apple.glb', 'carrot.glb', 'bowl.glb'],
    }

    for category_name, objects in categories.items():
        print(f"Rendering object category: {category_name}")
        try:
            env = CustomizableMujocoEnv(
                floor_texture='light_wood_v3.png',
                objects=objects,
                render_images=True,
                show_viewer=False,
                show_images=False
            )
            env.render_first_frame(
                output_dir=output_dir,
                prefix=f'objects_{category_name}',
                save_state=True
            )
            env.close()
            print(f"  ✓ Successfully rendered {category_name}\n")
        except Exception as e:
            print(f"  ✗ Failed to render {category_name}: {e}\n")


def render_random_variations(num_variations=5, output_dir='env_renders/random'):
    """Render random environment variations."""
    print(f"\n=== Rendering {num_variations} Random Variations ===\n")

    for i in range(num_variations):
        print(f"Rendering random variation {i+1}/{num_variations}")
        try:
            config = create_random_config(
                object_category='fruits',
                num_objects=3,
                headless=True
            )
            # Force rendering
            config['render_images'] = True

            env = CustomizableMujocoEnv(**config)
            env.render_first_frame(
                output_dir=output_dir,
                prefix=f'random_{i:03d}',
                save_state=True
            )
            env.close()
            print(f"  ✓ Successfully rendered variation {i+1}\n")
        except Exception as e:
            print(f"  ✗ Failed to render variation {i+1}: {e}\n")


def render_custom_scene(floor_texture, objects, output_dir, prefix):
    """Render a custom scene."""
    print(f"\n=== Rendering Custom Scene: {prefix} ===\n")

    try:
        # Parse objects if it's a comma-separated string
        if isinstance(objects, str):
            objects = [obj.strip() for obj in objects.split(',')]

        env = CustomizableMujocoEnv(
            floor_texture=floor_texture,
            objects=objects,
            render_images=True,
            show_viewer=False,
            show_images=False
        )
        env.render_first_frame(
            output_dir=output_dir,
            prefix=prefix,
            save_state=True
        )
        env.close()
        print(f"  ✓ Successfully rendered custom scene\n")
    except Exception as e:
        print(f"  ✗ Failed to render custom scene: {e}\n")


def create_gallery_html(output_dir='env_renders', gallery_name='Environment Gallery'):
    """Create an HTML gallery of all rendered images."""
    print(f"\n=== Creating HTML Gallery ===\n")

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{gallery_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section h2 {{
            color: #555;
            border-bottom: 2px solid #333;
            padding-bottom: 5px;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .gallery-item {{
            background: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .gallery-item img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .caption {{
            margin-top: 10px;
            text-align: center;
            font-weight: bold;
            color: #666;
        }}
        .info {{
            margin-top: 5px;
            font-size: 12px;
            color: #999;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>{gallery_name}</h1>
"""

    # Find all PNG files in the output directory
    sections = {}
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.png'):
                rel_path = os.path.relpath(root, output_dir)
                section_name = rel_path if rel_path != '.' else 'Main'

                if section_name not in sections:
                    sections[section_name] = []

                img_path = os.path.join(root, file)
                rel_img_path = os.path.relpath(img_path, output_dir)
                caption = file.replace('.png', '').replace('_', ' ').title()

                # Check for state file
                state_file = img_path.replace('.png', '_state.json').replace(
                    file.split('_')[-1].replace('.png', ''),
                    'state.json'
                )
                has_state = os.path.exists(state_file)

                sections[section_name].append({
                    'path': rel_img_path,
                    'caption': caption,
                    'has_state': has_state
                })

    # Generate HTML for each section
    for section_name, items in sorted(sections.items()):
        html_content += f"""
    <div class="section">
        <h2>{section_name.replace('_', ' ').title()}</h2>
        <div class="gallery">
"""
        for item in sorted(items, key=lambda x: x['caption']):
            state_info = '<br>State: Available' if item['has_state'] else ''
            html_content += f"""
            <div class="gallery-item">
                <img src="{item['path']}" alt="{item['caption']}">
                <div class="caption">{item['caption']}</div>
                <div class="info">Camera view{state_info}</div>
            </div>
"""
        html_content += """
        </div>
    </div>
"""

    html_content += """
</body>
</html>
"""

    gallery_path = os.path.join(output_dir, 'gallery.html')
    with open(gallery_path, 'w') as f:
        f.write(html_content)

    print(f"Gallery created at: {gallery_path}")
    print(f"Open with: firefox {gallery_path} (or your preferred browser)\n")


def main():
    parser = argparse.ArgumentParser(description='Batch render environment configurations')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'predefined', 'floors', 'objects', 'random', 'custom'],
                       help='Rendering mode')
    parser.add_argument('--output-dir', type=str, default='env_renders',
                       help='Base output directory for rendered images')
    parser.add_argument('--num-random', type=int, default=5,
                       help='Number of random variations to render (for random mode)')
    parser.add_argument('--create-gallery', action='store_true',
                       help='Create an HTML gallery of rendered images')

    # Custom scene options
    parser.add_argument('--floor-texture', type=str, default=None,
                       help='Floor texture for custom scene')
    parser.add_argument('--objects', type=str, default=None,
                       help='Comma-separated list of objects for custom scene')
    parser.add_argument('--prefix', type=str, default='custom',
                       help='Prefix for custom scene renders')

    args = parser.parse_args()

    print("="*60)
    print("Batch Environment Renderer")
    print("="*60)

    if args.mode == 'all':
        render_all_predefined_configs(os.path.join(args.output_dir, 'predefined'))
        render_all_floor_textures(os.path.join(args.output_dir, 'floors'))
        render_object_categories(os.path.join(args.output_dir, 'objects'))
        render_random_variations(args.num_random, os.path.join(args.output_dir, 'random'))

    elif args.mode == 'predefined':
        render_all_predefined_configs(os.path.join(args.output_dir, 'predefined'))

    elif args.mode == 'floors':
        render_all_floor_textures(os.path.join(args.output_dir, 'floors'))

    elif args.mode == 'objects':
        render_object_categories(os.path.join(args.output_dir, 'objects'))

    elif args.mode == 'random':
        render_random_variations(args.num_random, os.path.join(args.output_dir, 'random'))

    elif args.mode == 'custom':
        if not args.floor_texture or not args.objects:
            print("Error: --floor-texture and --objects are required for custom mode")
            return
        render_custom_scene(
            args.floor_texture,
            args.objects,
            args.output_dir,
            args.prefix
        )

    if args.create_gallery:
        create_gallery_html(args.output_dir)

    print("="*60)
    print("Batch rendering complete!")
    print("="*60)


if __name__ == '__main__':
    main()
