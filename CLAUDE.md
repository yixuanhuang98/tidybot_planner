
I want to refactor the codebase to align with standard Mujuco setup, where I want to partition the codebase as simulation environment and real environment and agent. The modification needs to have the following structure: 
env/
- assets
- mujuco 
- real
the environment needs to follow the standard step, observation, action feedback like typical mujuco environment, make sure all the messages passed are clearly defined similar to 
```
class Env:
def __init__(self, handler):
self.handler = handler
handler.launch()
def reset(self):
handler.set_states()
states = handler.get_states()
return get_observation(states), \
handler.get_extra()
def step(self, action):
handler.set_states(action=action)
handler.step()
states = handler.get_states()
return get_observation(states), \
get_reward(states), \
get_success(states) \
get_termination(states), \
get_time_out(states), \
handler.get_extra()
def render(self):
return handler.render()
def close(self):
handler.close()
```

agent/
- base_agent.py: this includes abstract implementation and helper function such as inverse kinematics 
- existing policies, each in an individual file

```

  - Environment and controller logic are tightly coupled
  - All policies are in a single file
  - No clear separation between simulation and real environments
  - Missing standard MuJoCo environment API (reset, step, observation/action spaces)

  Proposed Structure

  env/
  ├── __init__.py
  ├── base_env.py          # Abstract base environment class
  ├── assets/              # Move models/ directory here
  │   ├── kinova_gen3/
  │   ├── stanford_tidybot/
  │   └── meshes/
  ├── mujoco/
  │   ├── __init__.py
  │   ├── mujoco_env.py    # MuJoCo environment with standard API
  │   ├── mujoco_handler.py # MuJoCo-specific handler
  │   ├── controllers.py    # Simulation controllers
  │   └── renderer.py       # Rendering utilities
  └── real/
      ├── __init__.py
      ├── real_env.py       # Real robot environment
      ├── real_handler.py   # Real robot handler
      ├── arm_controller.py
      ├── base_controller.py
      └── servers/          # RPC servers
          ├── arm_server.py
          └── base_server.py

  agent/
  ├── __init__.py
  ├── base_agent.py         # Abstract agent with IK solver
  ├── teleop_policy.py
  ├── remote_policy.py
  ├── motion_planner_policy.py
  ├── go_to_cabinet_handle_policy.py
  └── go_to_cabinet_handle_policy_right.py

  Key Changes

  1. Environment API Standardization
    - Implement standard reset(), step(action), render(), close() methods
    - Define clear observation and action spaces
    - Add reward, success, termination, and timeout logic
  2. Handler Pattern
    - Create handler classes for MuJoCo and real environments
    - Handlers manage low-level operations (physics, hardware communication)
    - Environments provide high-level API
  3. Agent Modularization
    - Extract each policy into separate files
    - Create base agent with shared utilities (IK solver)
    - Maintain consistent interface across agents
  4. Asset Organization
    - Move MuJoCo models to env/assets/
    - Update all path references

  This refactoring will create a clean separation of concerns and make the codebase more maintainable and extensible.
```
