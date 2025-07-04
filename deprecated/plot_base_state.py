# Author: Jimmy Wu
# Date: October 2024

import math
import time
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless mode
import matplotlib.pyplot as plt
import numpy as np
import redis

ROBOT_WIDTH = 0.5  # Note: Approximate value

class Visualizer:
    def __init__(self, headless=True):
        self.headless = headless
        
        if not self.headless:
            # Odometry figure
            plt.figure('Odometry')
            plt.axis([-1, 1, -1, 1])
            self.fig_axis = plt.gca()
            self.fig_axis.set_aspect('equal')
            plt.grid(which='both')
            self.robot_line, = plt.plot([], color='tab:gray')
            self.robot_arrow = plt.arrow(0, 0, 0, 0, head_width=0)
            self.odom_line, = plt.plot([], '--', color='tab:blue')

            # Velocity figure
            plt.figure('Velocity')
            self.vel_fig_axis = plt.gca()
            self.vel_x_line, = plt.plot([], label='x')
            self.vel_y_line, = plt.plot([], label='y')
            self.vel_th_line, = plt.plot([], label='θ')
            plt.grid(True)
            plt.legend()

            # Bring to foreground
            plt.figure('Odometry')

    def draw(self, x, t_data, x_data, dx_data):
        if self.headless:
            # In headless mode, just print status instead of plotting
            print(f"Robot position: x={x[0]:.3f}, y={x[1]:.3f}, θ={x[2]:.3f} rad")
            if len(dx_data) > 0:
                latest_vel = dx_data[-1]
                print(f"Robot velocity: dx={latest_vel[0]:.3f}, dy={latest_vel[1]:.3f}, dθ={latest_vel[2]:.3f} rad/s")
            return
            
        # Robot outline
        th = x[2]
        angles = th + np.radians([135, 45, -45, -135], dtype=np.float32)
        corners = (math.sqrt(2) / 2) * np.stack((np.cos(angles, dtype=np.float32), np.sin(angles, dtype=np.float32)), axis=1)
        corners = np.array([x[0], x[1]], dtype=np.float32) + ROBOT_WIDTH * corners
        corners = np.append(corners, corners[:1], axis=0)
        self.robot_line.set_data(*corners.T)

        # Robot heading
        arrow_dx = 0.25 * ROBOT_WIDTH * math.cos(th)
        arrow_dy = 0.25 * ROBOT_WIDTH * math.sin(th)
        self.robot_arrow.set_data(x=x[0], y=x[1], dx=arrow_dx, dy=arrow_dy, head_width=ROBOT_WIDTH / 8)

        # Trajectory (odometry)
        x_data = np.array(x_data, dtype=np.float32)
        self.odom_line.set_data(x_data[:, 0], x_data[:, 1])

        # Robot velocity
        t_data = np.array(t_data, dtype=np.float64)  # Do not cast timestamps to np.float32, major loss of precision
        dx_data = np.array(dx_data, dtype=np.float32)
        self.vel_x_line.set_data(t_data, dx_data[:, 0])
        self.vel_y_line.set_data(t_data, dx_data[:, 1])
        self.vel_th_line.set_data(t_data, dx_data[:, 2])

        # Update plots
        self.fig_axis.relim()
        self.fig_axis.autoscale()
        self.vel_fig_axis.relim()
        self.vel_fig_axis.autoscale()
        plt.pause(0.001)

# Main visualization loop
visualizer = Visualizer(headless=True)  # Default to headless mode
r = redis.Redis()
x_data = deque(maxlen=100)
t_data = deque(maxlen=100)
dx_data = deque(maxlen=100)

print("Running base state visualization in headless mode.")
print("Robot position and velocity will be printed to console.")
print("To enable GUI plotting, change headless=True to headless=False in the code.")

while True:
    try:
        # Note: Redis format is 'x y th'
        x = np.array(r.get('x').decode().split(), dtype=np.float32)
        dx = np.array(r.get('dx').decode().split(), dtype=np.float32)
        t = time.time()
        x_data.append(x)
        t_data.append(t)
        dx_data.append(dx)
        visualizer.draw(x, t_data, x_data, dx_data)
    except:
        # Redis key does not exist
        pass
    time.sleep(0.1)
