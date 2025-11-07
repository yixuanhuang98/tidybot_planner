#!/usr/bin/env python3
"""
Quick test script for offline rendering functionality.
Tests that the customizable environment can load and render without errors.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_render():
    """Test basic rendering with minimal config."""
    print("Testing basic offline rendering...")

    try:
        from customizable_env import CustomizableMujocoEnv

        # Create environment with minimal config
        print("  Creating environment...")
        env = CustomizableMujocoEnv(
            floor_texture='light_wood_v3.png',
            objects=['apple.glb', 'banana.glb', 'tomato.glb'],
            render_images=True,
            show_viewer=False,
            show_images=False
        )

        print("  Environment created successfully!")

        # Try to render first frame
        print("  Rendering first frame...")
        images = env.render_first_frame(
            output_dir='test_renders',
            prefix='test',
            save_state=True
        )

        print(f"  Successfully rendered {len(images)} camera views!")
        for camera_name, img in images.items():
            print(f"    {camera_name}: {img.shape}")

        # Clean up
        env.close()
        print("  ✓ Test PASSED!\n")
        return True

    except Exception as e:
        print(f"  ✗ Test FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_texture():
    """Test with different floor texture."""
    print("Testing custom floor texture...")

    try:
        from customizable_env import CustomizableMujocoEnv

        print("  Creating environment with metal floor...")
        env = CustomizableMujocoEnv(
            floor_texture='metal.png',
            objects=['apple.glb'],
            render_images=False,  # Faster test without rendering
            show_viewer=False,
            show_images=False
        )

        print("  Environment created successfully!")

        # Just test reset
        print("  Testing reset...")
        env.reset()

        # Clean up
        env.close()
        print("  ✓ Test PASSED!\n")
        return True

    except Exception as e:
        print(f"  ✗ Test FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_xml_generation():
    """Test XML generation with absolute paths."""
    print("Testing XML generation...")

    try:
        from customizable_env import CustomizableMujocoEnv
        import xml.etree.ElementTree as ET

        print("  Creating environment...")
        env = CustomizableMujocoEnv(
            floor_texture='light_wood_v3.png',
            objects=['apple.glb', 'banana.glb'],
            render_images=False,
            show_viewer=False,
            show_images=False
        )

        # Check the generated XML file
        print(f"  Generated XML at: {env.mjcf_path}")

        # Parse and validate XML
        print("  Parsing XML...")
        tree = ET.parse(env.mjcf_path)
        root = tree.getroot()

        # Check include paths are absolute
        includes = root.findall('include')
        if includes:
            for include in includes:
                file_path = include.get('file')
                print(f"    Include path: {file_path}")
                if not os.path.isabs(file_path):
                    print(f"    ⚠ Warning: Include path is not absolute!")
                else:
                    print(f"    ✓ Include path is absolute")

        # Check asset file paths
        meshes = root.findall('.//mesh')
        for mesh in meshes:
            file_path = mesh.get('file')
            if file_path:
                print(f"    Mesh path: {file_path}")
                if not os.path.isabs(file_path):
                    print(f"    ⚠ Warning: Mesh path is not absolute!")
                else:
                    print(f"    ✓ Mesh path is absolute")

        # Check texture paths
        textures = root.findall('.//texture[@type="2d"]')
        for texture in textures:
            file_path = texture.get('file')
            if file_path:
                print(f"    Texture path: {file_path}")
                if not os.path.isabs(file_path):
                    print(f"    ⚠ Warning: Texture path is not absolute!")
                else:
                    print(f"    ✓ Texture path is absolute")

        # Clean up
        env.close()
        print("  ✓ Test PASSED!\n")
        return True

    except Exception as e:
        print(f"  ✗ Test FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Offline Rendering Test Suite")
    print("="*60 + "\n")

    results = []

    # Run tests
    results.append(("XML Generation", test_xml_generation()))
    results.append(("Custom Texture", test_custom_texture()))
    results.append(("Basic Render", test_basic_render()))

    # Print summary
    print("="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
