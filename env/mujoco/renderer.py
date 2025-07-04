"""
MuJoCo Renderer Abstraction
Handles different rendering backends for headless and visual environments
"""
import os
import time
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import mujoco

try:
    import cv2 as cv
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def setup_headless_rendering():
    """Set up headless rendering environment"""
    # Check if we already have a display
    if os.environ.get('DISPLAY'):
        print("Display already available, using existing X11 session")
        return 'x11'
    
    # Try to set up virtual framebuffer if available
    try:
        # Check if Xvfb is available
        result = subprocess.run(['which', 'Xvfb'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Xvfb detected - setting up virtual framebuffer for headless rendering")
            
            # Find an available display number
            display_num = 99
            while os.path.exists(f'/tmp/.X{display_num}-lock'):
                display_num += 1
            
            # Start virtual framebuffer
            os.environ['DISPLAY'] = f':{display_num}'
            xvfb_process = subprocess.Popen([
                'Xvfb', f':{display_num}', 
                '-screen', '0', '1024x768x24',
                '-ac', '+extension', 'GLX', '+render', '-noreset'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            time.sleep(3)  # Give Xvfb more time to start
            
            # Verify the display is working
            try:
                result = subprocess.run(['xdpyinfo', '-display', f':{display_num}'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"Virtual display :{display_num} started successfully")
                    return 'xvfb'
                else:
                    print("Failed to verify virtual display")
            except:
                print("Could not verify virtual display")
    except Exception as e:
        print(f"Xvfb setup failed: {e}")
    
    # Try EGL setup
    try:
        os.environ['MUJOCO_GL'] = 'egl'
        print("Using EGL backend for headless rendering")
        return 'egl'
    except:
        pass
    
    # Try OSMesa if available
    try:
        os.environ['MUJOCO_GL'] = 'osmesa'
        print("Using OSMesa backend for headless rendering")
        return 'osmesa'
    except:
        pass
    
    print("Warning: No suitable headless rendering backend found")
    return None


class BaseRenderer(ABC):
    """Abstract base class for MuJoCo renderers"""
    
    def __init__(self, model, data, width=640, height=480):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.rendering_working = False
    
    @abstractmethod
    def render_view(self, camera_config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Render a view with given camera configuration"""
        pass
    
    @abstractmethod
    def close(self):
        """Cleanup renderer resources"""
        pass


class MuJoCoOpenGLRenderer(BaseRenderer):
    """Real MuJoCo OpenGL renderer for headless environments"""
    
    def __init__(self, model, data, width=640, height=480):
        super().__init__(model, data, width, height)
        
        print("Setting up headless rendering...")
        # Set up headless rendering
        backend = setup_headless_rendering()
        if not backend:
            print("No suitable rendering backend available")
            raise RuntimeError("No suitable rendering backend available")
        
        print(f"Using backend: {backend}")
        
        try:
            print("Creating MuJoCo Renderer...")
            # Use the modern MuJoCo Renderer class which handles OpenGL context internally
            self.renderer = mujoco.Renderer(model, height=height, width=width)
            self.rendering_working = True
            
            print(f"Successfully initialized MuJoCo OpenGL renderer ({self.width}x{self.height})")
            
        except Exception as e:
            print(f"Failed to initialize MuJoCo OpenGL renderer: {e}")
            import traceback
            traceback.print_exc()
            self.rendering_working = False
            raise e
    
    def render_view(self, camera_config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Render actual MuJoCo camera view"""
        if not self.rendering_working:
            return self._render_error_placeholder()
        
        try:
            # Create camera
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, camera)
            
            # Set camera parameters
            camera.distance = camera_config['distance']
            camera.lookat = np.array(camera_config['lookat'])
            camera.elevation = camera_config['elevation']
            camera.azimuth = camera_config['azimuth']
            
            # Update scene and render
            self.renderer.update_scene(self.data, camera=camera)
            
            # Get the rendered image
            pixels = self.renderer.render()
            
            return pixels
            
        except Exception as e:
            print(f"Rendering error: {e}")
            self.rendering_working = False
            return self._render_error_placeholder()
    
    def _render_error_placeholder(self):
        """Generate error placeholder image"""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Red background to indicate error
        image[:, :, 0] = 100  # Red
        image[:, :, 1] = 30   # Green
        image[:, :, 2] = 30   # Blue
        
        # Add error text if OpenCV is available
        if HAS_OPENCV:
            try:
                font = cv.FONT_HERSHEY_SIMPLEX
                text = "RENDER ERROR"
                text_size = cv.getTextSize(text, font, 1, 2)[0]
                text_x = (self.width - text_size[0]) // 2
                text_y = (self.height + text_size[1]) // 2
                cv.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
            except:
                pass
        
        return image
    
    def close(self):
        """Cleanup rendering resources"""
        try:
            if hasattr(self, 'renderer') and self.renderer:
                self.renderer.close()
        except:
            pass


class PlaceholderRenderer(BaseRenderer):
    """Fallback renderer that generates placeholder images"""
    
    def __init__(self, model, data, width=640, height=480):
        super().__init__(model, data, width, height)
        self.rendering_working = True
        print(f"Using placeholder renderer ({self.width}x{self.height})")
    
    def render_view(self, camera_config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate placeholder image with view-specific patterns"""
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 50  # Dark gray background
        
        # Create view-specific patterns
        center_x, center_y = self.width // 2, self.height // 2
        
        # Extract view type from camera config
        view_type = self._get_view_type(camera_config)
        
        if view_type == 'base':
            # Base view - horizontal gradient
            for x in range(self.width):
                intensity = int(50 + (x / self.width) * 100)
                image[:, x, 0] = intensity  # Red channel
        elif view_type == 'wrist':
            # Wrist view - vertical gradient
            for y in range(self.height):
                intensity = int(50 + (y / self.height) * 100)
                image[y, :, 1] = intensity  # Green channel
        else:
            # Overview - radial gradient
            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    intensity = int(50 + min(100, dist / 3))
                    image[y, x, 2] = intensity  # Blue channel
        
        # Add timestamp and physics info if OpenCV is available
        if HAS_OPENCV:
            self._add_info_overlay(image, view_type)
        
        return image
    
    def _get_view_type(self, camera_config):
        """Determine view type from camera configuration"""
        # Simple heuristic based on camera distance and position
        distance = camera_config.get('distance', 2.0)
        elevation = camera_config.get('elevation', 0)
        
        if distance < 1.5:
            return 'wrist'
        elif distance > 2.5:
            return 'overview'
        else:
            return 'base'
    
    def _add_info_overlay(self, image, view_type):
        """Add informational overlay to the image"""
        try:
            font = cv.FONT_HERSHEY_SIMPLEX
            
            # Add view type label
            cv.putText(image, f"View: {view_type.upper()}", (10, 30), font, 0.7, (255, 255, 255), 2)
            
            # Add simulation info
            sim_time = f"Sim Time: {time.time():.1f}s"
            cv.putText(image, sim_time, (10, self.height - 50), font, 0.5, (200, 200, 200), 1)
            
            # Add placeholder notice
            cv.putText(image, "PLACEHOLDER", (10, self.height - 25), font, 0.5, (100, 100, 255), 1)
            
        except Exception:
            pass  # Silently fail if text rendering doesn't work
    
    def close(self):
        """Cleanup (nothing to cleanup for placeholder renderer)"""
        pass


class RendererFactory:
    """Factory for creating appropriate renderer based on environment"""
    
    @staticmethod
    def create_renderer(model, data, width=640, height=480, prefer_real=True):
        """Create the best available renderer"""
        if prefer_real:
            try:
                # Try to create real MuJoCo renderer
                renderer = MuJoCoOpenGLRenderer(model, data, width, height)
                if renderer.rendering_working:
                    return renderer
                else:
                    print("MuJoCo OpenGL renderer failed, falling back to placeholder")
            except Exception as e:
                print(f"Could not create MuJoCo OpenGL renderer: {e}")
        
        # Fall back to placeholder renderer
        return PlaceholderRenderer(model, data, width, height)


class MultiViewRenderer:
    """Manages multiple camera views for a scene"""
    
    def __init__(self, model, data, camera_views=None, width=640, height=480):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        
        # Default camera views if none provided
        if camera_views is None:
            self.camera_views = {
                'base': {'distance': 2.0, 'lookat': [0.5, 0.0, 0.5], 'elevation': -30, 'azimuth': 45},
                'wrist': {'distance': 1.0, 'lookat': [0.7, 0.0, 0.4], 'elevation': -45, 'azimuth': 90},
                'overview': {'distance': 3.0, 'lookat': [0.5, 0.0, 0.5], 'elevation': -60, 'azimuth': 0}
            }
        else:
            self.camera_views = camera_views
        
        # Create renderer
        self.renderer = RendererFactory.create_renderer(model, data, width, height)
        
        print(f"Initialized MultiViewRenderer with {len(self.camera_views)} views")
    
    def render_all_views(self) -> Dict[str, np.ndarray]:
        """Render all camera views and return as dictionary"""
        images = {}
        
        for view_name, view_config in self.camera_views.items():
            try:
                image = self.renderer.render_view(view_config)
                if image is not None:
                    images[view_name] = image
                else:
                    print(f"Failed to render view: {view_name}")
            except Exception as e:
                print(f"Error rendering view {view_name}: {e}")
        
        return images
    
    def close(self):
        """Cleanup renderer resources"""
        if self.renderer:
            self.renderer.close()