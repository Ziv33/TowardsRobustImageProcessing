"""
src/train_yolo.py

Exposes functional APIs to automate standard YOLOv8 training on clean data,
specific noise bands, mixed-distortion splits, and progressive curriculum stages.
Upgraded to use YOLOv8 Large with batch size memory management.
"""

import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import cv2
import torch
from src.distortions import apply_gaussian_noise, apply_jpeg_compression, apply_low_light

def fine_tune_on_distorted(
    epochs: int = 100, 
    distorted_ratio: float = 1.0, 
    intensity: str = "random",  # "low", "mid", "high", "random"
    base_model_path: str = "output/checkpoints/yolov8_clean.pt",
    output_name: str = "yolov8_finetuned"
) -> str:
    """
    Generates a targeted training split and fine-tunes a YOLOv8 model.
    """
    print(f"\n--- Creating Dataset Split (Ratio: {distorted_ratio:.2f}, Intensity: {intensity.upper()}) ---")
    
    current_file_dir = Path(__file__).resolve().parent
    project_root = current_file_dir.parent
    root_path = project_root / "data" / "kitti"
    
    clean_images_dir = root_path / "images/train"
    clean_labels_dir = root_path / "labels/train"
    
    # Establish distinct folder structures for different dataset variations
    split_name = f"split_{intensity}_{int(distorted_ratio * 100)}"
    dist_dir = root_path / "processed" / split_name
    distorted_images_dir = dist_dir / "images"
    distorted_labels_dir = dist_dir / "labels"
    
    distorted_images_dir.mkdir(parents=True, exist_ok=True)
    distorted_labels_dir.mkdir(parents=True, exist_ok=True)
    
    clean_image_paths = list(clean_images_dir.glob("*.png"))
    
    for img_path in clean_image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply corruption if ratio threshold is met
        if random.random() < distorted_ratio:
            distortion_type = random.choice(["Noise", "JPEG", "LowLight"])
            
            # Select range limits based on intensity configuration (aligned to validation extremes)
            if intensity == "low":
                sig_range = (0.0, 15.0)
                jpg_range = (70, 100)
                gam_range = (1.0, 1.5)
            elif intensity == "mid":
                sig_range = (15.0, 35.0)
                jpg_range = (30, 70)
                gam_range = (1.5, 3.0)
            elif intensity == "high":
                sig_range = (35.0, 60.0)
                jpg_range = (1, 30)
                gam_range = (3.0, 4.5)
            else:  # Random (full range matching validation sweeps)
                sig_range = (0.0, 60.0)
                jpg_range = (1, 100)
                gam_range = (1.0, 4.5)
                
            if distortion_type == "Noise":
                processed_rgb = apply_gaussian_noise(img_rgb, sigma=random.uniform(*sig_range))
            elif distortion_type == "JPEG":
                processed_rgb = apply_jpeg_compression(img_rgb, quality=random.randint(*jpg_range))
            elif distortion_type == "LowLight":
                processed_rgb = apply_low_light(img_rgb, gamma=random.uniform(*gam_range))
        else:
            processed_rgb = img_rgb.copy()
            
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(distorted_images_dir / img_path.name), processed_bgr)
        
        label_name = f"{img_path.stem}.txt"
        clean_lbl_path = clean_labels_dir / label_name
        if clean_lbl_path.exists():
            shutil.copy(clean_lbl_path, distorted_labels_dir / label_name)

    yaml_content = f"""path: {dist_dir.resolve()}
train: images
val: {root_path.resolve()}/images/val

names:
  0: car
  1: pedestrian
  2: cyclist
"""
    yaml_path = dist_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
        
    print(f"Dataset compiled at: {yaml_path}")
    
    # Clean fallback check (updated to YOLOv8 Large)
    if not Path(base_model_path).exists():
        print(f"Warning: Base weights '{base_model_path}' not found. Falling back to yolov8l.pt")
        base_model_path = "yolov8l.pt"
        
    return train_kitti_model(
        dataset_yaml=str(yaml_path),
        epochs=epochs,
        output_name=output_name,
        base_model=base_model_path
    )


def train_kitti_model(
    dataset_yaml: str,
    epochs: int = 100,
    imgsz: int = 960,
    output_name: str = "yolov8_clean",
    base_model: str = "yolov8l.pt"
) -> str:
    """
    Standard YOLOv8 model training loop wrapper.
    Optimized for memory usage (batch=32, workers=8).
    """
    print(f"Initializing training using base model {base_model}...")
    model = YOLO(base_model)
    device_id = "0" if torch.cuda.is_available() else "cpu"
    
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=32,  
        device=device_id,
        workers=8,
        verbose=True,
        patience=15,  # Early stopping threshold
    )
    
    runs_dir = Path("runs/detect")
    best_weights = sorted(list(runs_dir.glob("**/weights/best.pt")))
    
    if best_weights:
        current_file_dir = Path(__file__).resolve().parent
        project_root = current_file_dir.parent
        target_checkpoint = project_root / "output" / "checkpoints" / f"{output_name}.pt"
        target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(best_weights[-1], target_checkpoint)
        print(f"Weights successfully compiled at: {target_checkpoint}")
        return str(target_checkpoint)
    else:
        raise FileNotFoundError("Training completed, but weights file could not be located.")