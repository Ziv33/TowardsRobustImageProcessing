"""
src/tasks/registry.py

Provides a central registry and decorator to track task classes.
Safe to use with Jupyter %autoreload and importlib.reload.
"""

from typing import Dict, Type
from src.tasks.base_task import BaseTask

# Global dictionary to map task names to task classes
_TASK_REGISTRY: Dict[str, Type[BaseTask]] = {}

def register_task(name: str):
    """
    A class-level decorator used to register a new task with the framework.
    Safe to re-evaluate during module reloads.
    
    Args:
        name: Unique string identifier for the task.
    """
    def decorator(cls: Type[BaseTask]):
        if not issubclass(cls, BaseTask):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BaseTask to be registered.")
            
        # Check for existing registrations
        if name in _TASK_REGISTRY:
            # If the class name matches, allow overwriting to support Jupyter module reloads
            if _TASK_REGISTRY[name].__name__ == cls.__name__:
                _TASK_REGISTRY[name] = cls
                return cls
            raise KeyError(f"Task name '{name}' is already registered to a different class: {_TASK_REGISTRY[name].__name__}.")
            
        _TASK_REGISTRY[name] = cls
        return cls
    return decorator

def get_task_classes() -> Dict[str, Type[BaseTask]]:
    """
    Returns a copy of the registered task dictionary.
    """
    return dict(_TASK_REGISTRY)