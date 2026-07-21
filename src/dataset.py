"""
src/dataset.py

Directly downloads, extracts, splits, and normalizes the complete, official
12 GB KITTI Object Detection dataset. Implements the standard research split:
- Train: 3,712 images (000000.png to 003711.png)
- Val: 3,769 images (003712.png to 007480.png)
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from tqdm import tqdm

class KITTIDataset:
    """
    Manages the complete official KITTI dataset. Automatically downloads, extracts,
    splits, and normalizes coordinates if local folders are empty.
    """
    CLASS_MAPPING = {
        "car": 0,
        "van": 0,
        "truck": 0,
        "pedestrian": 1,
        "person_sitting": 1,
        "cyclist": 2,
    }

    def __init__(self, root_dir: str = "data/kitti", split: str = "val"):
        """
        Args:
            root_dir: Base directory to store and organize the KITTI dataset.
            split: Split folder to access ('train' or 'val').
        """
        self.root_path = Path(root_dir)
        self.split = split
        
        self.images_dir = self.root_path / "images" / split
        self.labels_dir = self.root_path / "labels" / split
        self.masks_dir = self.root_path / "masks" / split

        # Check if the folders contain processed data; if not, trigger the pipeline
        self.image_paths = sorted(list(self.images_dir.glob("*.png"))) if self.images_dir.exists() else []

        if not self.image_paths:
            print("--- Local KITTI dataset not found or incomplete ---")
            print("Beginning programmatic download of the full, official 12 GB dataset...")
            self._download_and_process_dataset()
            # Rediscover paths after processing completes
            self.image_paths = sorted(list(self.images_dir.glob("*.png")))
            
        print(f"Dataset split '{self.split}' initialized with {len(self.image_paths)} images.")

    def _download_file(self, url: str, dest_path: Path):
        """Downloads a file with an interactive real-time progress bar."""
        print(f"Downloading: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024  # 1 MB blocks
            
            with open(dest_path, "wb") as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=dest_path.name
            ) as bar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    bar.update(len(buffer))

    def _download_and_process_dataset(self):
        """Downloads official archives from average-kitti mirrors and processes them."""
        # Official KITTI AWS S3 direct links
        img_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
        lbl_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

        temp_dir = self.root_path / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        zip_img_path = temp_dir / "data_object_image_2.zip"
        zip_lbl_path = temp_dir / "data_object_label_2.zip"

        # 1. Download Archives if not already cached in temp
        if not zip_lbl_path.exists():
            self._download_file(lbl_url, zip_lbl_path)
        if not zip_img_path.exists():
            self._download_file(img_url, zip_img_path)

        # Create target directories for the splits
        for s in ["train", "val"]:
            (self.root_path / "images" / s).mkdir(parents=True, exist_ok=True)
            (self.root_path / "labels" / s).mkdir(parents=True, exist_ok=True)
            (self.root_path / "masks" / s).mkdir(parents=True, exist_ok=True)

        print("Extracting labels and images...")
        raw_labels_dir = temp_dir / "training" / "label_2"
        raw_images_dir = temp_dir / "training" / "image_2"

        # 2. Extract Labels
        if not raw_labels_dir.exists():
            with zipfile.ZipFile(zip_lbl_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

        # 3. Extract Images
        if not raw_images_dir.exists():
            with zipfile.ZipFile(zip_img_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

        print("Processing and organizing splits...")
        raw_img_paths = sorted(list(raw_images_dir.glob("*.png")))
        
        # Standard research division index limit
        split_limit = 3712 

        for idx, img_path in enumerate(tqdm(raw_img_paths, desc="Processing images")):
            # Determine split assignment based on index
            target_split = "train" if idx < split_limit else "val"
            
            dest_img_dir = self.root_path / "images" / target_split
            dest_lbl_dir = self.root_path / "labels" / target_split
            
            # Read image to obtain dynamic dimensions for label normalization
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Copy image directly to target split directory
            cv2.imwrite(str(dest_img_dir / img_path.name), img)

            # Process matching label
            raw_lbl_path = raw_labels_dir / f"{img_path.stem}.txt"
            yolo_labels = []
            
            if raw_lbl_path.exists():
                with open(raw_lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        class_name = parts[0].lower()
                        if class_name in self.CLASS_MAPPING:
                            class_id = self.CLASS_MAPPING[class_name]
                            
                            # Extract native KITTI absolute pixel coords [left, top, right, bottom]
                            left, top, right, bottom = map(float, parts[4:8])
                            
                            # Convert to normalized YOLO coordinate format [x_center, y_center, width, height]
                            box_w = right - left
                            box_h = bottom - top
                            x_center = left + (box_w / 2.0)
                            y_center = top + (box_h / 2.0)
                            
                            norm_x = x_center / w
                            norm_y = y_center / h
                            norm_w = box_w / w
                            norm_h = box_h / h
                            
                            yolo_labels.append(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")

            # Write normalized annotation file to target split directory
            with open(dest_lbl_dir / f"{img_path.stem}.txt", "w") as f_out:
                f_out.write("\n".join(yolo_labels))

        # 4. Clean up raw temporary zip files to preserve local storage space
        print("Cleaning up temporary directories...")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Complete original KITTI dataset processing successfully completed.")

    def __len__(self) -> int:
        return len(self.image_paths)
        
    def get_sample(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Retrieves a single sample by index."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds.")

        img_path = self.image_paths[index]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Load Segmentation Mask (defaults to empty values if missing)
        mask_path = self.masks_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
            
        # Parse labels
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append({
                            "class_id": int(parts[0]),
                            "bbox": [float(x) for x in parts[1:]]
                        })
                        
        metadata = {
            "filename": img_path.name,
            "mask": mask,
            "boxes": boxes
        }
        return image_rgb, metadata