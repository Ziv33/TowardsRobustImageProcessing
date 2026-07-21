"""
src/tasks/orb_matching.py

Low-level vision: ORB (Oriented FAST and Rotated BRIEF) feature extraction 
and keypoint matching. Evaluates keypoint preservation rates under degradation.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2
from src.tasks.base_task import BaseTask
from src.tasks.registry import register_task

@register_task("low_level_orb")
class ORBMatchingTask(BaseTask):
    """
    Evaluates the robustness of low-level local feature descriptors.
    Matches features extracted from degraded/restored images against 
    descriptors extracted from the clean baseline image.
    """
    
    def __init__(self, max_features: int = 400):
        """
        Args:
            max_features: Upper limit of keypoints to detect per frame.
        """
        self.orb = cv2.ORB_create(nfeatures=max_features)
        # BFMatcher with Hamming distance is the standard matcher for binary descriptors (ORB)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    @property
    def task_name(self) -> str:
        return "low_level_orb"
        
    def run(self, image: np.ndarray) -> Tuple[Any, Any]:
        """
        Detects keypoints and computes descriptors on the target image.
        
        Args:
            image: np.ndarray, RGB format (uint8)
            
        Returns:
            Tuple: (keypoints, descriptors)
                   - keypoints: List of cv2.KeyPoint objects
                   - descriptors: np.ndarray of binary feature vectors, shape (N, 32)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kps, descs = self.orb.detectAndCompute(gray, None)
        return kps, descs
        
    def evaluate(self, predictions: Tuple[Any, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Matches the computed descriptors against the baseline clean descriptors
        to calculate keypoint survival rate.
        
        Args:
            predictions: Tuple of (keypoints, descriptors) from current run.
            ground_truth: Dictionary containing 'baseline_descriptors'.
            
        Returns:
            Dict[str, float]: Metric dictionary containing:
                - "matching_ratio": Ratio of successfully matched keypoints relative
                                    to the original baseline count [0.0, 1.0].
        """
        _, target_desc = predictions
        baseline_desc = ground_truth.get("baseline_descriptors", None)
        
        # If either baseline or target contains zero features, matching ratio is zero
        if baseline_desc is None or target_desc is None or len(baseline_desc) == 0 or len(target_desc) == 0:
            return {"matching_ratio": 0.0}
            
        # Match descriptors (crossCheck=True ensures mutual matches)
        matches = self.bf.match(baseline_desc, target_desc)
        
        # The matching ratio is the number of valid matches divided by the
        # number of original features present in the clean baseline image
        matching_ratio = len(matches) / float(len(baseline_desc))
        
        return {"matching_ratio": min(1.0, matching_ratio)}