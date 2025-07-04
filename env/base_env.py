from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional


class BaseEnv(ABC):
    """Abstract base class for all environments following standard MuJoCo API"""
    
    def __init__(self, handler):
        self.handler = handler
        self.handler.launch()
    
    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        """Reset the environment and return initial observation and info"""
        self.handler.set_states()
        states = self.handler.get_states()
        return self.get_observation(states), self.handler.get_extra()
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, bool, Dict]:
        """Execute one environment step
        
        Returns:
            observation: Current observation
            reward: Reward for the action
            success: Whether the task was successful
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated (timeout)
            info: Additional information
        """
        self.handler.set_states(action=action)
        self.handler.step()
        states = self.handler.get_states()
        return (
            self.get_observation(states),
            self.get_reward(states),
            self.get_success(states),
            self.get_termination(states),
            self.get_timeout(states),
            self.handler.get_extra()
        )
    
    @abstractmethod
    def render(self) -> Any:
        """Render the environment"""
        return self.handler.render()
    
    @abstractmethod
    def close(self):
        """Close the environment and clean up resources"""
        self.handler.close()
    
    @abstractmethod
    def get_observation(self, states: Dict) -> Any:
        """Extract observation from states"""
        pass
    
    @abstractmethod
    def get_reward(self, states: Dict) -> float:
        """Calculate reward from states"""
        pass
    
    @abstractmethod
    def get_success(self, states: Dict) -> bool:
        """Check if task was successful"""
        pass
    
    @abstractmethod
    def get_termination(self, states: Dict) -> bool:
        """Check if episode should terminate"""
        pass
    
    @abstractmethod
    def get_timeout(self, states: Dict) -> bool:
        """Check if episode timed out"""
        pass
    
    @property
    @abstractmethod
    def observation_space(self):
        """Define the observation space"""
        pass
    
    @property
    @abstractmethod
    def action_space(self):
        """Define the action space"""
        pass