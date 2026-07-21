"""
src/tasks/base_task.py

Defines the unified abstract base class for all vision pipeline tasks.
Ensures that low-level, mid-level, and high-level vision tasks expose the 
exact same programming interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np

class BaseTask(ABC):
    """
    Abstract Base Class representing a generic vision task.
    All tasks in the robustness framework must implement these methods.
    """
    
    @property
    @abstractmethod
    def task_name(self) -> str:
        """
        Returns the registered name of the task as a string.
        """
        pass
        
    @abstractmethod
    def run(self, image: np.ndarray) -> Any:
        """
        Executes the main algorithm or model inference on the input image.
        
        Args:
            image: np.ndarray, RGB format (uint8)
            
        Returns:
            Any: Task-specific raw predictions (e.g., keypoints, segmentation mask, bounding boxes).
        """
        pass
        
    @abstractmethod
    def evaluate(self, predictions: Any, ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates task predictions against ground-truth labels.
        
        Args:
            predictions: The raw output returned by self.run().
            ground_truth: Dictionary containing ground-truth metadata.
            
        Returns:
            Dict[str, float]: Mapping of metric names to calculated float scores.
        """
        pass