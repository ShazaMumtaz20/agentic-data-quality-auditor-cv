"""
Data Loader Module
Loads images from ImageFolder-style directory structure (one folder per class).
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
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
