"""
Comprehensive Preprocessing Pipeline
Handles full dataset preprocessing: resize, fix, augment, and save.
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from data_loader import DataLoader
from fixer import ImageFixer
from quality_metrics import QualityMetrics


class DatasetPreprocessor:
    """
    Comprehensive preprocessing pipeline for ML-ready datasets.
    Applies all fixes, resizing, and augmentation.
    """
    
    def __init__(self, data_loader: DataLoader, 
                 output_dir: str = "cleaned_dataset",
                 removed_dir: str = "removed_images",
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            data_loader: DataLoader instance
            output_dir: Directory to save cleaned images
            removed_dir: Directory to save removed images
            target_size: Target size for resizing (width, height)
        """
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.removed_dir = Path(removed_dir)
        self.target_size = target_size
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.removed_dir.mkdir(exist_ok=True, parents=True)
        
        self.fixer = ImageFixer(data_loader, output_dir=str(output_dir))
        self.preprocessing_log = []
        
    def preprocess_dataset(self,
                          blur_scores: List[float],
                          noise_scores: List[float],
                          brightness_scores: List[float],
                          contrast_scores: List[float],
                          saturation_scores: List[float],
                          corruption_flags: List[bool],
                          image_paths: List[str],
                          class_distribution: Dict[str, int],
                          agent_actions: Dict) -> Dict:
        """
        Comprehensive preprocessing pipeline.
        
        Args:
            blur_scores: List of blur scores
            noise_scores: List of noise scores
            brightness_scores: List of brightness scores
            contrast_scores: List of contrast scores
            saturation_scores: List of saturation scores
            corruption_flags: List of corruption flags
            image_paths: List of image paths
            class_distribution: Class distribution dictionary
            agent_actions: Agent decisions dictionary
            
        Returns:
            Dictionary with preprocessing statistics
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE PREPROCESSING PIPELINE")
        print("="*80)
        
        stats = {
            'total_images': len(image_paths),
            'processed': 0,
            'removed': 0,
            'fixed': 0,
            'augmented': 0,
            'kept_unchanged': 0,
            'removed_reasons': {}
        }
        
        # Get excluded indices from agent
        excluded_indices = set()
        if 'excluded_images' in agent_actions:
            excluded_indices = set([ex['image_index'] for ex in agent_actions['excluded_images']])
        
        # Get class labels
        class_labels = []
        for img_path in image_paths:
            path_obj = Path(img_path)
            if len(path_obj.parts) >= 2:
                class_labels.append(path_obj.parts[-2])
            else:
                class_labels.append('unknown')
        
        # Step 1: Remove corrupted images
        print("\nStep 1: Removing corrupted images...")
        for i, (img_path, is_corrupted) in enumerate(zip(image_paths, corruption_flags)):
            if is_corrupted:
                self._move_to_removed(img_path, "Corrupted image")
                excluded_indices.add(i)
                stats['removed'] += 1
                stats['removed_reasons']['corruption'] = stats['removed_reasons'].get('corruption', 0) + 1
        
        # Step 2: Process each image
        print(f"\nStep 2: Preprocessing {len(image_paths)} images...")
        print(f"  Target size: {self.target_size[0]}x{self.target_size[1]}")
        
        for i, img_path in enumerate(image_paths):
            if i in excluded_indices:
                continue
            
            try:
                # Load image
                image = self.data_loader.load_image(img_path)
                
                # Determine what preprocessing to apply
                needs_denoising = False
                needs_brightness = False
                needs_contrast = False
                needs_saturation = False
                
                # Check noise
                if i < len(noise_scores) and noise_scores[i] > 20.0:
                    needs_denoising = True
                
                # Check brightness
                if i < len(brightness_scores):
                    if brightness_scores[i] < 50.0 or brightness_scores[i] > 200.0:
                        needs_brightness = True
                
                # Check contrast
                if i < len(contrast_scores) and contrast_scores[i] < 30.0:
                    needs_contrast = True
                
                # Check saturation
                if i < len(saturation_scores) and saturation_scores[i] > 200.0:
                    needs_saturation = True  # Over-saturated, reduce
                
                # Apply preprocessing pipeline
                processed_image = self.fixer.preprocess_image(
                    image,
                    target_size=self.target_size,
                    apply_resize=True,
                    apply_denoising=needs_denoising,
                    apply_brightness_norm=needs_brightness,
                    apply_contrast_enhance=needs_contrast,
                    apply_saturation_adjust=needs_saturation,
                    target_brightness=128.0,
                    denoise_method='bilateral',
                    contrast_method='CLAHE',
                    saturation_factor=0.9 if needs_saturation else 1.0
                )
                
                # Save processed image
                self._save_processed_image(img_path, processed_image)
                
                fixes_applied = []
                if needs_denoising:
                    fixes_applied.append("denoise")
                if needs_brightness:
                    fixes_applied.append("brightness_norm")
                if needs_contrast:
                    fixes_applied.append("contrast_enhance")
                if needs_saturation:
                    fixes_applied.append("saturation_adjust")
                
                if fixes_applied:
                    stats['fixed'] += 1
                    self.preprocessing_log.append({
                        'image': img_path,
                        'action': 'fixed',
                        'fixes': fixes_applied
                    })
                else:
                    stats['kept_unchanged'] += 1
                    self.preprocessing_log.append({
                        'image': img_path,
                        'action': 'kept',
                        'fixes': []
                    })
                
                stats['processed'] += 1
                
                if stats['processed'] % 10 == 0:
                    print(f"  Processed {stats['processed']}/{len(image_paths) - stats['removed']} images...")
                    
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                self._move_to_removed(img_path, f"Processing error: {e}")
                stats['removed'] += 1
                continue
        
        # Step 3: Handle class imbalance with augmentation
        print(f"\nStep 3: Handling class imbalance with augmentation...")
        stats['augmented'] = self._augment_imbalanced_classes(
            class_distribution, class_labels, image_paths, excluded_indices
        )
        
        print(f"\n{'='*80}")
        print("PREPROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total images: {stats['total_images']}")
        print(f"Processed: {stats['processed']}")
        print(f"Fixed: {stats['fixed']}")
        print(f"Augmented: {stats['augmented']}")
        print(f"Removed: {stats['removed']}")
        print(f"Kept unchanged: {stats['kept_unchanged']}")
        
        return stats
    
    def _save_processed_image(self, original_path: str, processed_image: np.ndarray):
        """Save processed image preserving directory structure."""
        path_obj = Path(original_path)
        relative_path = path_obj.relative_to(self.data_loader.dataset_path)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for saving
        processed_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), processed_bgr)
    
    def _move_to_removed(self, image_path: str, reason: str):
        """Move image to removed_images directory."""
        path_obj = Path(image_path)
        relative_path = path_obj.relative_to(self.data_loader.dataset_path)
        removed_path = self.removed_dir / relative_path
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(str(path_obj), str(removed_path))
        except Exception as e:
            # If move fails, try copy and delete
            shutil.copy2(str(path_obj), str(removed_path))
            path_obj.unlink()
    
    def _augment_imbalanced_classes(self,
                                   class_distribution: Dict[str, int],
                                   class_labels: List[str],
                                   image_paths: List[str],
                                   excluded_indices: set) -> int:
        """Augment under-represented classes."""
        # Calculate target count (average or max)
        class_counts = {}
        for i, img_path in enumerate(image_paths):
            if i in excluded_indices:
                continue
            class_name = class_labels[i]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if len(class_counts) == 0:
            return 0
        
        avg_count = sum(class_counts.values()) / len(class_counts)
        target_count = int(avg_count * 1.2)  # 20% above average
        
        augmented_count = 0
        
        for class_name, count in class_counts.items():
            if count < avg_count * 0.7:  # Under-represented
                needed = target_count - count
                if needed > 0:
                    # Get images from this class
                    class_images = [i for i, label in enumerate(class_labels) 
                                  if label == class_name and i not in excluded_indices]
                    
                    if len(class_images) > 0:
                        # Augment images
                        for _ in range(min(needed, len(class_images) * 2)):  # Limit augmentation
                            import random
                            source_idx = random.choice(class_images)
                            source_path = image_paths[source_idx]
                            
                            try:
                                image = self.data_loader.load_image(source_path)
                                
                                # Apply random augmentation
                                augmented = self.fixer.augment_image(image, augmentation_type='random')
                                
                                # Resize and save
                                processed = self.fixer.preprocess_image(
                                    augmented,
                                    target_size=self.target_size,
                                    apply_resize=True,
                                    apply_denoising=False,
                                    apply_brightness_norm=False,
                                    apply_contrast_enhance=False,
                                    apply_saturation_adjust=False
                                )
                                
                                # Generate unique augmented filename
                                path_obj = Path(source_path)
                                aug_filename = f"{path_obj.stem}_aug_{augmented_count}{path_obj.suffix}"
                                aug_path = path_obj.parent / aug_filename
                                relative_path = aug_path.relative_to(self.data_loader.dataset_path)
                                output_path = self.output_dir / relative_path
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(str(output_path), processed_bgr)
                                
                                augmented_count += 1
                                
                            except Exception as e:
                                print(f"  Warning: Could not augment {source_path}: {e}")
                                continue
        
        return augmented_count
