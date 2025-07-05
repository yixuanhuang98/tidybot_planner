# MuJoCo Web Renderer

A real-time web-based renderer for MuJoCo simulations that displays multiple camera views in a browser interface.

## Features

- **Real-time streaming**: Live updates at 30 FPS
- **Multiple camera views**: Base, wrist, and overview perspectives
- **Modern web interface**: Beautiful, responsive design with controls
- **WebSocket communication**: Low-latency real-time updates
- **Headless support**: Works in environments without displays
- **Fallback rendering**: Graceful degradation to placeholder images

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements_web.txt
   ```

2. **Verify MuJoCo installation**:
   ```bash
   python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
   ```

## Usage

### Basic Usage

```python
from env.mujoco.web_renderer import create_web_renderer
import mujoco

# Load your MuJoCo model
model = mujoco.MjModel.from_xml_file("your_model.xml")
data = mujoco.MjData(model)

# Create web renderer
web_renderer = create_web_renderer(model, data, width=640, height=480, port=5000)

# Start the web server
web_renderer.run(host='0.0.0.0', debug=False)
```

### Run Demo

```bash
cd env/mujoco
python demo_web_renderer.py
```

Then open your browser to: `http://localhost:5000`

### Integration with Your Environment

```python
import threading
from env.mujoco.web_renderer import create_web_renderer

class YourEnvironment:
    def __init__(self):
        # ... your environment setup ...
        
        # Add web renderer
        self.web_renderer = create_web_renderer(
            self.model, 
            self.data, 
            width=640, 
            height=480, 
            port=5000
        )
        
        # Start web server in background
        self.web_thread = threading.Thread(
            target=self.web_renderer.run, 
            kwargs={'host': '0.0.0.0', 'debug': False},
            daemon=True
        )
        self.web_thread.start()
    
    def step(self, action):
        # ... your step logic ...
        
        # Simulation data is automatically updated in the web renderer
        # No additional code needed!
        
        return observation, reward, done, info
```

## Web Interface

The web interface provides:

- **Live camera feeds**: Multiple simultaneous views
- **Control buttons**: Start/stop streaming
- **Real-time statistics**: FPS, frame count, runtime
- **Status indicators**: Connection status
- **Responsive design**: Works on desktop and mobile

### Camera Views

1. **Base View**: Main perspective of the scene
2. **Wrist View**: Close-up view, typically from robot's perspective
3. **Overview View**: Wide-angle view of the entire scene

### Controls

- **Start Streaming**: Begin real-time updates
- **Stop Streaming**: Pause updates
- **Auto-refresh**: Automatic reconnection on disconnect

## Configuration

### Custom Camera Views

```python
custom_views = {
    'front': {'distance': 2.0, 'lookat': [0, 0, 0], 'elevation': 0, 'azimuth': 0},
    'side': {'distance': 2.0, 'lookat': [0, 0, 0], 'elevation': 0, 'azimuth': 90},
    'top': {'distance': 2.0, 'lookat': [0, 0, 0], 'elevation': -90, 'azimuth': 0},
}

web_renderer = WebRenderer(model, data, camera_views=custom_views)
```

### Render Settings

```python
web_renderer = create_web_renderer(
    model, 
    data,
    width=1280,      # Image width
    height=720,      # Image height
    port=8080,       # Web server port
)

# Adjust frame rate
web_renderer.fps = 60  # Target FPS
```

## Troubleshooting

### Common Issues

1. **"No suitable rendering backend"**
   - Install Xvfb: `sudo apt-get install xvfb`
   - Or use EGL/OSMesa for headless rendering

2. **"Port already in use"**
   - Change port: `create_web_renderer(..., port=5001)`
   - Or kill existing process: `sudo lsof -t -i:5000 | xargs kill`

3. **"Images not updating"**
   - Check browser console for WebSocket errors
   - Ensure firewall allows the port
   - Try refreshing the page

### Performance Tips

- **Reduce image size** for better performance: `width=320, height=240`
- **Lower frame rate** for slower connections: `web_renderer.fps = 15`
- **Close unused browser tabs** to free up resources
- **Use wired connection** for stable streaming

## API Reference

### WebRenderer Class

```python
class WebRenderer:
    def __init__(self, model, data, width=640, height=480, port=5000)
    def run(self, host='0.0.0.0', debug=False)
    def start_streaming(self)
    def stop_streaming(self)
    def close(self)
```

### HTTP Endpoints

- `GET /` - Main web interface
- `GET /api/views` - Get available camera views
- `GET /api/start_stream` - Start streaming
- `GET /api/stop_stream` - Stop streaming
- `GET /api/stats` - Get rendering statistics

### WebSocket Events

- `connect` - Client connected
- `disconnect` - Client disconnected
- `frame_update` - New frame available
- `status` - Server status update

## License

This web renderer is part of the MuJoCo environment tools and follows the same licensing terms as the main project.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your MuJoCo and dependency versions
3. Test with the demo script first
4. Check browser console for JavaScript errors 