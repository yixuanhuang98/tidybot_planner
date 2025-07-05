"""
Real-time Web Renderer for MuJoCo Simulator
Displays multiple camera views in a web browser with live updates
"""
import os
import time
import base64
import threading
from io import BytesIO
from flask import Flask, render_template, jsonify, Response, request
from flask_socketio import SocketIO, emit
import numpy as np
from PIL import Image
import json

from .renderer import MultiViewRenderer, RendererFactory

try:
    import cv2 as cv
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class WebRenderer:
    """Web-based renderer that streams MuJoCo views to a browser"""
    
    def __init__(self, model, data, width=640, height=480, port=5000):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.port = port
        
        # Initialize renderer
        self.renderer = MultiViewRenderer(model, data, width=width, height=height)
        
        # Flask app setup
        self.app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
        self.app.config['SECRET_KEY'] = 'mujoco_renderer_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # State management
        self.is_streaming = False
        self.stream_thread = None
        self.current_images = {}
        self.fps = 30
        self.frame_count = 0
        self.start_time = time.time()
        
        # Setup routes
        self._setup_routes()
        
        print(f"WebRenderer initialized on port {port}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page with camera views"""
            view_names = list(self.renderer.camera_views.keys())
            return render_template('renderer.html', 
                                 view_names=view_names,
                                 width=self.width,
                                 height=self.height)
        
        @self.app.route('/api/views')
        def get_views():
            """Get available camera views"""
            return jsonify({
                'views': list(self.renderer.camera_views.keys()),
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'streaming': self.is_streaming
            })
        
        @self.app.route('/api/start_stream')
        def start_stream():
            """Start image streaming"""
            if not self.is_streaming:
                self.start_streaming()
            return jsonify({'status': 'started', 'streaming': self.is_streaming})
        
        @self.app.route('/api/stop_stream')
        def stop_stream():
            """Stop image streaming"""
            if self.is_streaming:
                self.stop_streaming()
            return jsonify({'status': 'stopped', 'streaming': self.is_streaming})
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get rendering statistics"""
            runtime = time.time() - self.start_time
            current_fps = self.frame_count / runtime if runtime > 0 else 0
            return jsonify({
                'frames_rendered': self.frame_count,
                'runtime_seconds': runtime,
                'current_fps': current_fps,
                'target_fps': self.fps,
                'is_streaming': self.is_streaming
            })
        
        # SocketIO events
        @self.socketio.on('connect')
        def handle_connect():
            print(f"Client connected: {request.sid}")
            emit('status', {'connected': True, 'streaming': self.is_streaming})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_frame')
        def handle_frame_request():
            """Send current frame to client"""
            if self.current_images:
                self._emit_frame_update()
    
    def _numpy_to_base64(self, image_array):
        """Convert numpy array to base64 encoded image"""
        try:
            # Convert BGR to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # MuJoCo typically returns RGB, but just in case
                image_rgb = image_array
            else:
                image_rgb = image_array
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb.astype(np.uint8))
            
            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None
    
    def _emit_frame_update(self):
        """Emit frame update to all connected clients"""
        if not self.current_images:
            return
        
        # Convert images to base64
        image_data = {}
        for view_name, image in self.current_images.items():
            base64_image = self._numpy_to_base64(image)
            if base64_image:
                image_data[view_name] = base64_image
        
        if image_data:
            # Emit to all connected clients
            self.socketio.emit('frame_update', {
                'images': image_data,
                'timestamp': time.time(),
                'frame_count': self.frame_count
            })
    
    def _stream_loop(self):
        """Main streaming loop"""
        print("Starting streaming loop...")
        frame_time = 1.0 / self.fps
        
        while self.is_streaming:
            loop_start = time.time()
            
            try:
                # Render all views
                images = self.renderer.render_all_views()
                
                if images:
                    self.current_images = images
                    self.frame_count += 1
                    
                    # Emit frame update
                    self._emit_frame_update()
                
                # Control frame rate
                loop_duration = time.time() - loop_start
                sleep_time = max(0, frame_time - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error in streaming loop: {e}")
                time.sleep(0.1)  # Brief pause before retrying
        
        print("Streaming loop stopped")
    
    def start_streaming(self):
        """Start the streaming thread"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        
        print("Streaming started")
    
    def stop_streaming(self):
        """Stop the streaming thread"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Wait for thread to finish
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
        
        print("Streaming stopped")
    
    def run(self, host='0.0.0.0', debug=False):
        """Run the web server"""
        try:
            print(f"Starting web server on http://{host}:{self.port}")
            print("Press Ctrl+C to stop the server")
            
            # Start streaming by default
            self.start_streaming()
            
            # Run the Flask app
            self.socketio.run(self.app, host=host, port=self.port, debug=debug)
            
        except KeyboardInterrupt:
            print("\nShutting down web server...")
        finally:
            self.stop_streaming()
            self.renderer.close()
    
    def close(self):
        """Cleanup resources"""
        self.stop_streaming()
        self.renderer.close()


def create_web_renderer(model, data, width=640, height=480, port=5000):
    """Factory function to create a web renderer"""
    return WebRenderer(model, data, width, height, port)


if __name__ == "__main__":
    # Example usage - would need actual MuJoCo model and data
    print("WebRenderer module - import and use create_web_renderer() function") 