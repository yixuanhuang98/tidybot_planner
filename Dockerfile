# Use NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_CACHE=/app/.cache/huggingface
ENV WANDB_CACHE_DIR=/app/.cache/wandb

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libboost-all-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone and install lerobot from git (do this early for better caching)
RUN git clone https://github.com/huggingface/lerobot.git /tmp/lerobot
RUN cd /tmp/lerobot && pip install -e .

# Install additional Python dependencies not included in base image or lerobot
RUN pip install --no-cache-dir \
    mujoco==3.2.4 \
    mujoco-py==2.1.2.14 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    flask==3.1.1 \
    flask-socketio==5.3.6 \
    glfw==2.9.0 \
    gym==0.26.2 \
    dm-env==1.6 \
    dm-tree==0.1.9 \
    jax==0.6.2 \
    jaxlib==0.6.2 \
    jaxlie==1.5.0 \
    jaxtyping==0.3.2 \
    eigenpy==3.5.1 \
    hpp-fcl==2.4.4 \
    pin==2.7.0 \
    robot-descriptions==1.18.0 \
    trimesh==4.6.12 \
    yourdfpy==0.0.58 \
    viser==0.2.23 \
    tyro==0.9.24 \
    evdev==1.9.2 \
    pyopengl==3.1.9 \
    redis==5.0.6 \
    colorlog==6.9.0 \
    loguru==0.7.3 \
    inquirerpy==0.3.4 \
    rich==14.0.0 \ 
    opencv-python-headless==4.5.4.58 \
    ruckig

# Set working directory
WORKDIR /app

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/wandb

# Create volume mount point for project files
VOLUME ["/app"]

# Expose any necessary ports (adjust as needed)
EXPOSE 8000

# Set the default command (can be overridden at runtime)
CMD ["python", "main.py", "--sim", "--motion_planner_table_stack", "--save-images"]