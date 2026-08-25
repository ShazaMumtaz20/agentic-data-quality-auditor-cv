"""
Data Loader Module
Loads images from ImageFolder-style directory structure (one folder per class).
"""

import os
import cv2
import json
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path


class DataLoader:
    """
    Loads image dataset from ImageFolder structure.
    Assumes structure: dataset_root/class_name/image.jpg
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the data loader.
        
        Args:
            dataset_path: Path to the root directory containing class folders
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        self.classes = []
        self.image_paths = []
        self.class_labels = []
        
    def load_dataset(self) -> Dict[str, List[str]]:
        """
        Load all images from the dataset.
        
        Returns:
            Dictionary mapping class names to lists of image paths
        """
        # Rebuild cached state so repeated loads do not duplicate paths or classes.
        self.classes = []
        self.image_paths = []
        self.class_labels = []
        dataset_dict = {}
        
        # Get all subdirectories (classes)
        for class_dir in sorted(self.dataset_path.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.classes.append(class_name)
                
                # Get all image files in this class directory (case-insensitive)
                # Collect all files and filter by case-insensitive extension
                valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                image_files = []
                for file_path in class_dir.iterdir():
                    if file_path.is_file():
                        # Check extension case-insensitively
                        ext = file_path.suffix.lower()
                        if ext in valid_extensions:
                            image_files.append(file_path)
                
                # Remove duplicates (in case filesystem is case-insensitive)
                image_files = list(set(image_files))
                
                dataset_dict[class_name] = [str(img) for img in sorted(image_files)]
                
                # Store paths and labels for easy access
                for img_path in dataset_dict[class_name]:
                    self.image_paths.append(img_path)
                    self.class_labels.append(class_name)
        
        if len(self.classes) == 0:
            raise ValueError(f"No class directories found in {self.dataset_path}")
        
        return dataset_dict
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load a single image as RGB numpy array.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array (H, W, 3) in RGB format
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the number of images per class.
        
        Returns:
            Dictionary mapping class names to image counts
        """
        distribution = {}
        for class_name in self.classes:
            class_dir = self.dataset_path / class_name
            # Use case-insensitive matching and remove duplicates
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = []
            for file_path in class_dir.iterdir():
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in valid_extensions:
                        image_files.append(file_path)
            image_files = list(set(image_files))  # Remove duplicates
            distribution[class_name] = len(image_files)
        return distribution
    
    def get_total_images(self) -> int:
        """Get total number of images in the dataset."""
        return len(self.image_paths)

    def build_fixed_split(
        self,
        seed: int = 42,
        train_ratio: float = 0.7,
        calibration_ratio: float = 0.15,
        output_dir: str = "results",
    ) -> Dict[str, Any]:
        """
        Build a deterministic split using image identity paths.

        Requirements from the refactoring specification:
        - fixed random seed default = 42
        - configurable via CLI in later stage wiring
        - zero identity overlap across train_pool / calibration_set / held_out_test_set
        - print assertion results and sample counts
        - save split info to results/split_info.json
        """
        if not self.image_paths:
            self.load_dataset()

        all_paths = list(self.image_paths)
        rng = random.Random(seed)
        shuffled = all_paths[:]
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_count = int(total * train_ratio)
        calibration_count = int(total * calibration_ratio)
        test_count = total - train_count - calibration_count

        train_pool = shuffled[:train_count]
        calibration_set = shuffled[train_count:train_count + calibration_count]
        held_out_test_set = shuffled[train_count + calibration_count:train_count + calibration_count + test_count]

        train_set = set(train_pool)
        calibration_set_set = set(calibration_set)
        test_set = set(held_out_test_set)

        overlap_train_cal = len(train_set & calibration_set_set)
        overlap_train_test = len(train_set & test_set)
        overlap_cal_test = len(calibration_set_set & test_set)

        print("Dataset split assertions:")
        print(f"  train_pool vs calibration_set overlap: {overlap_train_cal}")
        print(f"  train_pool vs held_out_test_set overlap: {overlap_train_test}")
        print(f"  calibration_set vs held_out_test_set overlap: {overlap_cal_test}")
        print(f"  train_pool count: {len(train_pool)}")
        print(f"  calibration_set count: {len(calibration_set)}")
        print(f"  held_out_test_set count: {len(held_out_test_set)}")

        assert overlap_train_cal == 0, "train_pool and calibration_set overlap detected"
        assert overlap_train_test == 0, "train_pool and held_out_test_set overlap detected"
        assert overlap_cal_test == 0, "calibration_set and held_out_test_set overlap detected"

        split_info = {
            "seed": seed,
            "train_ratio": train_ratio,
            "calibration_ratio": calibration_ratio,
            "counts": {
                "train_pool": len(train_pool),
                "calibration_set": len(calibration_set),
                "held_out_test_set": len(held_out_test_set),
            },
            "train_pool": train_pool,
            "calibration_set": calibration_set,
            "held_out_test_set": held_out_test_set,
            "assertions": {
                "train_vs_calibration_overlap": overlap_train_cal,
                "train_vs_test_overlap": overlap_train_test,
                "calibration_vs_test_overlap": overlap_cal_test,
            },
        }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "split_info.json", "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2)

        print(f"Split information saved to {output_path / 'split_info.json'}")
        return split_info

    def get_split_summary(self, split_info: Dict[str, Any]) -> Dict[str, int]:
        """Return the sample count summary for a split_info dictionary."""
        return {
            "train_pool": len(split_info.get("train_pool", [])),
            "calibration_set": len(split_info.get("calibration_set", [])),
            "held_out_test_set": len(split_info.get("held_out_test_set", [])),
        }
