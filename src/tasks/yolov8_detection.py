"""
src/tasks/yolov8_detection.py

High-level vision: Object Detection using YOLOv8.
Evaluates both Precision and Recall natively across all classes.
"""

from typing import Dict, Any, List
import numpy as np
from ultralytics import YOLO
from src.tasks.base_task import BaseTask
from src.tasks.registry import register_task
from src.evaluation import compute_bbox_iou

@register_task("high_level_yolo")
class YOLOv8DetectionTask(BaseTask):
    """
    Object detector utilizing a native KITTI-trained YOLOv8 model.
    Runs inference at a matched resolution to prevent scale mismatches.
    """
    
    def __init__(self, model_path: str = "output/checkpoints/yolov8_clean.pt", imgsz: int = 960, conf: float = 0.15):
        """
        Args:
            model_path: Path to the KITTI-trained weights.
            imgsz: Inference image size.
            conf: Prediction confidence threshold (default 0.15).
        """
        import torch
        self.model = YOLO(model_path)

        # Run on GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')

        self.imgsz = imgsz
        self.conf = conf 

    @property
    def task_name(self) -> str:
        return "high_level_yolo"
        
    def run(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection inference at the matched resolution.
        """
        # Pass the configured confidence threshold to the predict call
        results = self.model.predict(image, verbose=False, imgsz=self.imgsz, conf=self.conf)[0]
        predictions = []
        h, w = image.shape[:2]
        
        for box in results.boxes:
            predictions.append({
                "class_id": int(box.cls[0].cpu().numpy()),  # Class ID (0: Car, 1: Pedestrian, 2: Cyclist)
                "bbox": [
                    float(box.xywh[0][0] / w),
                    float(box.xywh[0][1] / h),
                    float(box.xywh[0][2] / w),
                    float(box.xywh[0][3] / h)
                ],
                "confidence": float(box.conf[0].cpu().numpy())
            })
            
        return predictions
        
    def evaluate(self, predictions: List[Dict[str, Any]], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes precision and recall rates split per individual class (Car=0, Pedestrian=1, Cyclist=2).
        """
        gt_boxes = ground_truth.get("boxes", [])
        class_targets = [0, 1, 2]
        metrics = {}
        
        total_recalls = []
        total_precisions = []
        
        for target_class in class_targets:
            gt_class_boxes = [g for g in gt_boxes if g["class_id"] == target_class]
            pred_class_boxes = [p for p in predictions if p["class_id"] == target_class]
            
            if not gt_class_boxes:
                continue
                
            matched_gt_indices = set()
            matched_pred_indices = set()
            
            # Map predictions to ground-truth boxes using a calibrated IoU threshold of 0.35
            for p_idx, pred in enumerate(pred_class_boxes):
                p_box = pred["bbox"]
                for gt_idx, gt in enumerate(gt_class_boxes):
                    if gt_idx in matched_gt_indices:
                        continue
                        
                    if compute_bbox_iou(p_box, gt["bbox"]) >= 0.35:
                        matched_gt_indices.add(gt_idx)
                        matched_pred_indices.add(p_idx)
                        break
            
            # Precision and Recall Calculations
            recall = len(matched_gt_indices) / float(len(gt_class_boxes)) if len(gt_class_boxes) > 0 else 0.0
            precision = len(matched_pred_indices) / float(len(pred_class_boxes)) if len(pred_class_boxes) > 0 else 1.0
            
            metrics[f"recall_class_{target_class}"] = recall
            metrics[f"precision_class_{target_class}"] = precision
            
            total_recalls.append(recall)
            total_precisions.append(precision)
            
        metrics["mean_recall"] = float(np.mean(total_recalls)) if total_recalls else 0.0
        metrics["mean_precision"] = float(np.mean(total_precisions)) if total_precisions else 1.0
        return metrics