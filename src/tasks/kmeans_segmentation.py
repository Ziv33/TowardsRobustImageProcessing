"""
src/tasks/kmeans_segmentation.py

Mid-level vision: Vectorized unsupervised pixel clustering in RGB color space
using K-Means. Evaluates semantic area preservation under degradation.
"""

from typing import Dict, Any
import numpy as np
import cv2
from src.tasks.base_task import BaseTask
from src.tasks.registry import register_task
from src.evaluation import compute_segmentation_iou

@register_task("mid_level_kmeans")
class KMeansSegmentationTask(BaseTask):
    """
    Groups pixels into K color-space clusters. Compares resulting boundaries 
    against ground-truth semantic partitions using Intersection over Union (IoU).
    """
    
    def __init__(self, k: int = 4):
        """
        Args:
            k: Number of target color clusters (standard class divisions: 4).
        """
        self.k = k
        # Standard convergence criteria for OpenCV K-Means execution
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
    @property
    def task_name(self) -> str:
        return "mid_level_kmeans"
        
    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Runs color-space clustering to segment pixels.
        
        Args:
            image: np.ndarray, RGB format (uint8)
            
        Returns:
            np.ndarray: Single-channel label mask (H, W), values range [0, k-1]
        """
        # Flatten image into a list of RGB pixel vectors
        pixels = image.reshape((-1, 3)).astype(np.float32)
        
        # Execute K-Means clustering (attempts = 10 for initialization robustness)
        _, labels, _ = cv2.kmeans(
            pixels, self.k, None, self.criteria, 
            attempts=10, flags=cv2.KMEANS_PP_CENTERS
        )
        
        # Reshape label array back to original spatial layout
        segmented_mask = labels.reshape((image.shape[0], image.shape[1])).astype(np.uint8)
        return segmented_mask
        
    def evaluate(self, predictions: np.ndarray, ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates predictions against semantic ground truth. Aligns unsupervised
        clusters to ground-truth IDs based on maximum overlap, then computes IoU.
        
        Args:
            predictions: np.ndarray, segmented cluster mask (H, W).
            ground_truth: Dict containing "mask" (ground-truth array).
            
        Returns:
            Dict[str, float]: Metric dictionary containing:
                - "mean_iou": Average IoU across all aligned classes.
                - "iou_class_{id}": Individual aligned IoU scores.
        """
        gt_mask = ground_truth.get("mask", None)
        if gt_mask is None:
            return {"mean_iou": 0.0}
            
        pred_mask = predictions
        aligned_mask = np.zeros_like(pred_mask)
        
        # Map unsupervised cluster IDs to ground-truth class IDs.
        # Since K-Means does not know what a "car" or "road" is, we map each cluster 
        # to the ground-truth ID that has the highest pixel overlap.
        for cluster_id in range(self.k):
            cluster_pixels = (pred_mask == cluster_id)
            if not np.any(cluster_pixels):
                continue
                
            # Count the ground-truth classes overlapping with this cluster
            overlapping_gt_labels = gt_mask[cluster_pixels]
            best_gt_class = int(np.bincount(overlapping_gt_labels).argmax())
            
            aligned_mask[cluster_pixels] = best_gt_class
            
        # Compute standard IoU on the aligned semantic mask
        per_class_iou = compute_segmentation_iou(aligned_mask, gt_mask, num_classes=4)
        mean_iou = float(np.mean(list(per_class_iou.values())))
        
        metrics = {"mean_iou": mean_iou}
        for class_id, iou_val in per_class_iou.items():
            metrics[f"iou_class_{class_id}"] = iou_val
            
        return metrics