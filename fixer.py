"""
Auto-Fix Module
Applies safe fixes to images: denoising, brightness normalization, and sharpening.
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from data_loader import DataLoader


class ImageFixer:
    """
    Applies safe fixes to improve image quality based on agent decisions.
    """
    
    def __init__(self, data_loader: DataLoader, output_dir: str = "cleaned_dataset"):
        """
        Initialize the image fixer.
        
        Args:
            data_loader: DataLoader instance
            output_dir: Directory to save fixed images (default: cleaned_dataset)
        """
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.applied_fixes = []  # Track what fixes were applied
        
    def denoise_image(self, image: np.ndarray, method: str = 'fastNlMeans') -> np.ndarray:
        """
        Apply denoising to an image.
        
        Args:
            image: RGB image array (H, W, 3)
            method: Denoising method ('fastNlMeans', 'bilateral', 'gaussian', or 'median')
            
        Returns:
            Denoised image
        """
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if method == 'fastNlMeans':
            # Fast Non-Local Means Denoising (best for blur)
            denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
        elif method == 'bilateral':
            # Bilateral Filter (preserves edges, good for noise)
            denoised = cv2.bilateralFilter(img_bgr, 9, 75, 75)
        elif method == 'gaussian':
            # Gaussian blur (mild denoising)
            denoised = cv2.GaussianBlur(img_bgr, (5, 5), 0)
        elif method == 'median':
            # Median filter (good for salt-and-pepper noise)
            denoised = cv2.medianBlur(img_bgr, 5)
        else:
            raise ValueError(f"Unknown denoising method: {method}. Use 'fastNlMeans', 'bilateral', 'gaussian', or 'median'")
        
        # Convert back to RGB
        denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        return denoised_rgb
    
    def sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Apply mild sharpening to an image.
        
        Args:
            image: RGB image array (H, W, 3)
            strength: Sharpening strength (0.0 to 2.0, default 1.0)
            
        Returns:
            Sharpened image
        """
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength
        
        # Apply kernel to each channel
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Blend with original to avoid over-sharpening
        result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def normalize_brightness(self, image: np.ndarray, target_brightness: float = 128.0) -> np.ndarray:
        """
        Normalize image brightness to target value.
        
        Args:
            image: RGB image array (H, W, 3)
            target_brightness: Target mean brightness (0-255)
            
        Returns:
            Brightness-normalized image
        """
        # Convert to grayscale to compute current brightness
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        current_brightness = np.mean(gray)
        
        if current_brightness == 0:
            return image  # Avoid division by zero
        
        # Compute adjustment factor
        adjustment = target_brightness / current_brightness
        
        # Apply adjustment to RGB image
        normalized = image.astype(np.float32) * adjustment
        
        # Clip to valid range [0, 255]
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        return normalized
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Resize image to target size while preserving aspect ratio using padding.
        
        Args:
            image: RGB image array (H, W, 3)
            target_size: Target (width, height) tuple
            
        Returns:
            Resized image with padding
        """
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        # Calculate scaling factor to fit image in target size
        scale = min(target_w / w, target_h / h)
        
        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding offsets (center the image)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'CLAHE') -> np.ndarray:
        """
        Enhance image contrast using CLAHE or histogram equalization.
        
        Args:
            image: RGB image array (H, W, 3)
            method: 'CLAHE' or 'histogram_eq'
            
        Returns:
            Contrast-enhanced image
        """
        if method == 'CLAHE':
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            return enhanced
        
        elif method == 'histogram_eq':
            # Convert to YUV, apply histogram equalization to Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            return enhanced
        
        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")
    
    def adjust_saturation(self, image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Adjust image saturation.
        
        Args:
            image: RGB image array (H, W, 3)
            factor: Saturation factor (1.0 = no change, >1.0 = increase, <1.0 = decrease)
            
        Returns:
            Saturation-adjusted image
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Adjust saturation channel
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        
        # Clip to valid range
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to RGB
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return adjusted
    
    def augment_image(self, image: np.ndarray, augmentation_type: str = 'random') -> np.ndarray:
        """
        Apply data augmentation to image.
        
        Args:
            image: RGB image array (H, W, 3)
            augmentation_type: Type of augmentation ('flip_h', 'flip_v', 'rotate', 'brightness_jitter', 'random')
            
        Returns:
            Augmented image
        """
        import random
        
        if augmentation_type == 'random':
            # Randomly choose an augmentation
            aug_type = random.choice(['flip_h', 'flip_v', 'rotate', 'brightness_jitter'])
        else:
            aug_type = augmentation_type
        
        if aug_type == 'flip_h':
            return cv2.flip(image, 1)  # Horizontal flip
        elif aug_type == 'flip_v':
            return cv2.flip(image, 0)  # Vertical flip
        elif aug_type == 'rotate':
            # Random rotation between -15 and 15 degrees
            angle = random.uniform(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            return rotated
        elif aug_type == 'brightness_jitter':
            # Slight brightness adjustment
            factor = random.uniform(0.9, 1.1)
            jittered = (image.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
            return jittered
        else:
            return image
    
    def preprocess_image(self, image: np.ndarray,
                        target_size: Tuple[int, int] = (224, 224),
                        apply_resize: bool = True,
                        apply_denoising: bool = False,
                        apply_brightness_norm: bool = False,
                        apply_contrast_enhance: bool = False,
                        apply_saturation_adjust: bool = False,
                        target_brightness: float = 128.0,
                        denoise_method: str = 'bilateral',
                        contrast_method: str = 'CLAHE',
                        saturation_factor: float = 1.0) -> np.ndarray:
        """
        Comprehensive preprocessing pipeline for a single image.
        Applies all fixes in the correct order.
        
        Args:
            image: RGB image array (H, W, 3)
            target_size: Target size for resizing (width, height)
            apply_resize: Whether to resize to target size
            apply_denoising: Whether to apply denoising
            apply_brightness_norm: Whether to normalize brightness
            apply_contrast_enhance: Whether to enhance contrast
            apply_saturation_adjust: Whether to adjust saturation
            target_brightness: Target brightness for normalization
            denoise_method: Denoising method to use
            contrast_method: Contrast enhancement method ('CLAHE' or 'histogram_eq')
            saturation_factor: Saturation adjustment factor
            
        Returns:
            Preprocessed image
        """
        # Step 1: Resize first (if needed) - do this early to speed up other operations
        if apply_resize:
            image = self.resize_image(image, target_size=target_size)
        
        # Step 2: Denoising (before other enhancements)
        if apply_denoising:
            image = self.denoise_image(image, method=denoise_method)
        
        # Step 3: Brightness normalization
        if apply_brightness_norm:
            image = self.normalize_brightness(image, target_brightness=target_brightness)
        
        # Step 4: Contrast enhancement
        if apply_contrast_enhance:
            image = self.enhance_contrast(image, method=contrast_method)
        
        # Step 5: Saturation adjustment (last, as it affects color)
        if apply_saturation_adjust:
            image = self.adjust_saturation(image, factor=saturation_factor)
        
        return image
    
    def fix_image(self, image_path: str, apply_denoising: bool = False,
                  apply_brightness_norm: bool = False,
                  apply_sharpening: bool = False,
                  target_brightness: float = 128.0,
                  denoise_method: str = 'fastNlMeans',
                  sharpen_strength: float = 1.0) -> np.ndarray:
        """
        Apply fixes to a single image (legacy method for backward compatibility).
        
        Args:
            image_path: Path to the image
            apply_denoising: Whether to apply denoising
            apply_brightness_norm: Whether to normalize brightness
            apply_sharpening: Whether to apply mild sharpening
            target_brightness: Target brightness for normalization
            denoise_method: Denoising method to use
            sharpen_strength: Sharpening strength (0.0 to 2.0)
            
        Returns:
            Fixed image
        """
        # Load image
        image = self.data_loader.load_image(image_path)
        
        # Apply denoising if requested
        if apply_denoising:
            image = self.denoise_image(image, method=denoise_method)
        
        # Apply brightness normalization if requested
        if apply_brightness_norm:
            image = self.normalize_brightness(image, target_brightness=target_brightness)
        
        # Apply sharpening if requested (after denoising to restore some detail)
        if apply_sharpening:
            image = self.sharpen_image(image, strength=sharpen_strength)
        
        return image
    
    def apply_agent_actions(self, actions: List[Dict], image_paths: List[str]) -> Tuple[int, int, int, List[str]]:
        """
        Apply fixes based on agent decisions.
        
        Args:
            actions: List of action dictionaries from agent.decide_actions()
            image_paths: List of all image paths
            
        Returns:
            Tuple of (fixed_count, skipped_count, excluded_count, applied_fixes_log)
        """
        fixed_count = 0
        skipped_count = 0
        excluded_count = 0
        applied_fixes_log = []
        
        # Track which images need which fixes
        image_fixes = {}  # {image_index: [list of fixes to apply]}
        
        for action in actions:
            action_type = action['type']
            indices = action['indices']
            reason = action.get('reason', '')
            
            for idx in indices:
                if idx not in image_fixes:
                    image_fixes[idx] = []
                
                if action_type == 'exclude':
                    # Don't add to image_fixes - these will be excluded
                    pass
                elif action_type == 'denoise':
                    image_fixes[idx].append({
                        'type': 'denoise',
                        'method': action.get('method', 'fastNlMeans'),
                        'reason': reason
                    })
                elif action_type == 'brightness_normalize':
                    image_fixes[idx].append({
                        'type': 'brightness_normalize',
                        'target_brightness': action.get('target_brightness', 128.0),
                        'reason': reason
                    })
        
        # Get excluded indices
        excluded_indices = set()
        for action in actions:
            if action['type'] == 'exclude':
                excluded_indices.update(action.get('indices', []))
        
        print(f"\nApplying fixes based on agent decisions...")
        print(f"  Excluding {len(excluded_indices)} images (irreversible defects)")
        print(f"  Fixing {len(image_fixes)} images...")
        
        for idx, fixes in image_fixes.items():
            if idx in excluded_indices:
                continue  # Skip excluded images
            if idx >= len(image_paths):
                continue
                
            try:
                img_path = image_paths[idx]
                
                # Determine what fixes to apply
                apply_denoising = False
                denoise_method = 'fastNlMeans'
                apply_brightness = False
                target_brightness = 128.0
                apply_sharpening = False
                
                fix_descriptions = []
                for fix in fixes:
                    if fix['type'] == 'denoise':
                        apply_denoising = True
                        denoise_method = fix.get('method', 'fastNlMeans')
                        fix_descriptions.append(f"denoise({denoise_method})")
                    elif fix['type'] == 'brightness_normalize':
                        apply_brightness = True
                        target_brightness = fix.get('target_brightness', 128.0)
                        fix_descriptions.append(f"brightness({target_brightness})")
                
                # Apply fixes
                fixed_image = self.fix_image(
                    img_path,
                    apply_denoising=apply_denoising,
                    apply_brightness_norm=apply_brightness,
                    apply_sharpening=apply_sharpening,
                    target_brightness=target_brightness,
                    denoise_method=denoise_method
                )
                
                # Save fixed image (preserve directory structure)
                img_path_obj = Path(img_path)
                relative_path = img_path_obj.relative_to(self.data_loader.dataset_path)
                output_path = self.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert RGB to BGR for saving
                fixed_bgr = cv2.cvtColor(fixed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), fixed_bgr)
                
                fixed_count += 1
                applied_fixes_log.append(f"Fixed {img_path}: {', '.join(fix_descriptions)}")
                
                if fixed_count % 10 == 0:
                    print(f"  Fixed {fixed_count} images...")
                    
            except Exception as e:
                print(f"  Warning: Could not fix {image_paths[idx]}: {e}")
                skipped_count += 1
                continue
        
        # Also copy images that don't need fixes (to preserve dataset structure)
        # But exclude those marked for exclusion
        print("Copying images that don't need fixes...")
        all_indices = set(range(len(image_paths)))
        fixed_indices = set(image_fixes.keys())
        unchanged_indices = all_indices - fixed_indices - excluded_indices
        
        for idx in unchanged_indices:
            try:
                img_path = image_paths[idx]
                img_path_obj = Path(img_path)
                relative_path = img_path_obj.relative_to(self.data_loader.dataset_path)
                output_path = self.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy original image
                shutil.copy2(img_path, output_path)
                skipped_count += 1  # Count as "skipped" (no fix needed)
            except Exception as e:
                print(f"  Warning: Could not copy {image_paths[idx]}: {e}")
                continue
        
        excluded_count = len(excluded_indices)
        print(f"Fix complete! Fixed {fixed_count} images, copied {len(unchanged_indices)} unchanged images, excluded {excluded_count} images.")
        return fixed_count, skipped_count, excluded_count, applied_fixes_log
    
    def fix_dataset(self, blur_scores: List[float],
                   noise_scores: List[float],
                   brightness_scores: List[float],
                   image_paths: List[str],
                   blur_threshold: float = 100.0,
                   noise_threshold: float = 20.0,
                   brightness_dark_threshold: float = 50.0,
                   brightness_bright_threshold: float = 200.0,
                   target_brightness: float = 128.0) -> Tuple[int, int]:
        """
        Fix images in the dataset based on quality metrics.
        
        Args:
            blur_scores: List of blur scores for each image
            noise_scores: List of noise scores for each image
            brightness_scores: List of brightness scores for each image
            image_paths: List of paths to images
            blur_threshold: Threshold below which images are considered blurry
            noise_threshold: Threshold above which images are considered noisy
            brightness_dark_threshold: Threshold below which images are too dark
            brightness_bright_threshold: Threshold above which images are too bright
            target_brightness: Target brightness for normalization
            
        Returns:
            Tuple of (number of images fixed, number of images skipped)
        """
        fixed_count = 0
        skipped_count = 0
        
        print("\nApplying fixes to images...")
        
        for i, (img_path, blur, noise, brightness) in enumerate(
            zip(image_paths, blur_scores, noise_scores, brightness_scores)
        ):
            try:
                # Determine what fixes to apply
                needs_denoising = blur < blur_threshold or noise > noise_threshold
                needs_brightness = brightness < brightness_dark_threshold or brightness > brightness_bright_threshold
                
                if not (needs_denoising or needs_brightness):
                    skipped_count += 1
                    continue
                
                # Apply fixes
                fixed_image = self.fix_image(
                    img_path,
                    apply_denoising=needs_denoising,
                    apply_brightness_norm=needs_brightness,
                    target_brightness=target_brightness
                )
                
                # Save fixed image (preserve directory structure)
                img_path_obj = Path(img_path)
                relative_path = img_path_obj.relative_to(self.data_loader.dataset_path)
                output_path = self.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert RGB to BGR for saving
                fixed_bgr = cv2.cvtColor(fixed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), fixed_bgr)
                
                fixed_count += 1
                
                if fixed_count % 10 == 0:
                    print(f"  Fixed {fixed_count} images...")
                    
            except Exception as e:
                print(f"  Warning: Could not fix {img_path}: {e}")
                skipped_count += 1
                continue
        
        print(f"Fix complete! Fixed {fixed_count} images, skipped {skipped_count} images.")
        return fixed_count, skipped_count
