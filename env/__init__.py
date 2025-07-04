from .base_env import BaseEnv
from .state import EnvironmentState, RobotState, ObjectState, Observation, Action

# Optional imports that may fail due to dependencies
try:
    from .mujoco import AbstractMujocoEnv, BlocksEnv, CabinetEnv, DrawerEnv
    _mujoco_available = True
except ImportError as e:
    print(f"Warning: MuJoCo environments not available: {e}")
    _mujoco_available = False

try:
    from .real.real_env import RealEnv
    _real_available = True
except ImportError as e:
    print(f"Warning: Real environment not available: {e}")
    _real_available = False

# Base exports (always available)
__all__ = [
    'BaseEnv',
    'EnvironmentState', 'RobotState', 'ObjectState', 'Observation', 'Action'
]

# Add optional exports
if _mujoco_available:
    __all__.extend(['AbstractMujocoEnv', 'BlocksEnv', 'CabinetEnv', 'DrawerEnv'])

if _real_available:
    __all__.extend(['RealEnv'])