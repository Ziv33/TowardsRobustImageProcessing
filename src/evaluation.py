"""
src/evaluation.py

Provides standard performance and alignment metrics for robustness evaluation.
All calculations use high-precision float64 operations to avoid division-by-zero
errors and numerical underflow.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

def compute_snr(clean: np.ndarray, distorted: np.ndarray) -> float:
    """
    Computes the Signal-to-Noise Ratio (SNR) in decibels (dB) between a clean 
    reference image and a distorted target image.
    
    Args:
        clean: np.ndarray, original clean image (uint8)
        distorted: np.ndarray, degraded target image (uint8)
        
    Returns:
        float: SNR value in dB. Higher values indicate less distortion.
               Returns infinity (float('inf')) if the images are identical.
    """
    # Cast to float64 to prevent integer overflow during squaring operations
    c_64 = clean.astype(np.float64)
    d_64 = distorted.astype(np.float64)
    
    # Calculate signal power (mean of squared clean pixel intensities)
    signal_power = np.mean(c_64 ** 2)
    
    # Calculate noise power (mean squared difference between clean and distorted)
    noise_power = np.mean((c_64 - d_64) ** 2)
    
    if noise_power == 0.0:
        return float('inf')
        
    # Standard logarithmic ratio calculation
    return 10.0 * np.log10(signal_power / noise_power)


def compute_segmentation_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> Dict[int, float]:
    """
    Computes the Intersection over Union (IoU) metric for each individual class 
    index in a semantic segmentation mask.
    
    Args:
        pred_mask: np.ndarray, predicted label array (H, W)
        gt_mask: np.ndarray, ground-truth label array (H, W)
        num_classes: Total number of classes to evaluate.
        
    Returns:
        Dict[int, float]: Mapping of class IDs to their corresponding IoU score [0.0, 1.0].
    """
    ious = {}
    for c in range(num_classes):
        pred_c = (pred_mask == c)
        gt_c = (gt_mask == c)
        
        intersection = np.sum(pred_c & gt_c)
        union = np.sum(pred_c | gt_c)
        
        if union == 0:
            # If the class is completely absent from both prediction and ground truth,
            # we count it as a perfect match (IoU = 1.0)
            ious[c] = 1.0
        else:
            ious[c] = float(intersection) / float(union)
            
    return ious


def compute_bbox_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Computes the Intersection over Union (IoU) score between two normalized
    bounding boxes using YOLO format [x_center, y_center, width, height].
    
    Args:
        boxA: List[float], [x, y, w, h] normalized coordinates
        boxB: List[float], [x, y, w, h] normalized coordinates
        
    Returns:
        float: IoU value ranging from 0.0 (no overlap) to 1.0 (perfect alignment).
    """
    # Convert center-based YOLO format back to absolute corners [x1, y1, x2, y2]
    ax1, ay1 = boxA[0] - boxA[2] / 2.0, boxA[1] - boxA[3] / 2.0
    ax2, ay2 = boxA[0] + boxA[2] / 2.0, boxA[1] + boxA[3] / 2.0
    
    bx1, by1 = boxB[0] - boxB[2] / 2.0, boxB[1] - boxB[3] / 2.0
    bx2, by2 = boxB[0] + boxB[2] / 2.0, boxB[1] + boxB[3] / 2.0
    
    # Calculate coordinates of the intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    
    # Calculate intersection area
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection_area = inter_w * inter_h
    
    # Calculate individual box areas
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    
    # Calculate union area
    union_area = areaA + areaB - intersection_area
    
    if union_area <= 0.0:
        return 0.0
        
    return float(intersection_area / union_area)