"""
Auto-Fix Module
Applies safe fixes to images: denoising, brightness normalization, and sharpening.
"""

import cv2
import numpy as np
import shutil
import random
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
            # Fast Non-Local Means Denoising (increased strength for better noise removal)
            denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, 20, 20, 7, 21)
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
        Normalize image brightness to target value using adaptive adjustment.
        
        Args:
            image: RGB image array (H, W, 3)
            target_brightness: Target mean brightness (0-255)
            
        Returns:
            Brightness-normalized image
        """
        # Convert to grayscale to compute current brightness
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        current_brightness = np.mean(gray)
        
        if current_brightness == 0:
            return image  # Avoid division by zero
        
        # Calculate the difference
        brightness_diff = target_brightness - current_brightness
        
        # Use adaptive adjustment based on how far from target
        if abs(brightness_diff) < 5:
            return image  # Already close to target, no adjustment needed
        
        # For extreme cases, be more aggressive
        if current_brightness < 30:  # Very dark
            adjustment_factor = 0.85  # Move 85% toward target
            min_adj, max_adj = 1.2, 4.0  # Allow significant increase
        elif current_brightness > 220:  # Very bright
            adjustment_factor = 0.75  # Move 75% toward target
            min_adj, max_adj = 0.4, 0.9  # Allow significant decrease
        elif current_brightness < 50:  # Dark
            adjustment_factor = 0.80  # Move 80% toward target
            min_adj, max_adj = 1.1, 2.5
        elif current_brightness > 200:  # Bright
            adjustment_factor = 0.70  # Move 70% toward target
            min_adj, max_adj = 0.5, 0.95
        else:  # Moderate brightness
            adjustment_factor = 0.75  # Move 75% toward target
            min_adj, max_adj = 0.7, 1.4
        
        # Calculate target brightness after adjustment
        target_adjusted = current_brightness + (brightness_diff * adjustment_factor)
        
        # Compute adjustment factor
        adjustment = target_adjusted / current_brightness
        
        # Apply limits
        adjustment = np.clip(adjustment, min_adj, max_adj)
        
        # Apply adjustment to RGB image
        normalized = image.astype(np.float32) * adjustment
        
        # Clip to valid range [0, 255]
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        return normalized
    
    def normalize_sharpness(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize over-sharpened images using Gaussian smoothing.
        Uses stronger smoothing for better effect.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Sharpness-normalized image
        """
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Apply stronger Gaussian smoothing to reduce sharpness (larger kernel, higher sigma)
        smoothed = cv2.GaussianBlur(img_bgr, (9, 9), 2.0)
        
        # Blend with original (75% smoothed, 25% original) for stronger effect
        blended = cv2.addWeighted(smoothed, 0.75, img_bgr, 0.25, 0)
        
        # Convert back to RGB
        smoothed_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        
        return smoothed_rgb
    
    def reduce_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce high contrast by compressing the dynamic range more effectively.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Contrast-reduced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        # Get current L channel statistics
        l_min = np.min(l)
        l_max = np.max(l)
        l_mean = np.mean(l)
        l_std = np.std(l)
        
        # More aggressive compression for high contrast images
        if l_std > 50:  # Very high contrast
            compression_factor = 0.5  # Compress to 50% of original range
            blend_ratio = 0.8  # Use 80% of compressed version
        elif l_std > 40:  # High contrast
            compression_factor = 0.6  # Compress to 60% of original range
            blend_ratio = 0.75  # Use 75% of compressed version
        else:  # Moderate contrast
            compression_factor = 0.7  # Compress to 70% of original range
            blend_ratio = 0.7  # Use 70% of compressed version
        
        l_range = l_max - l_min
        if l_range < 1e-6:
            return image  # No contrast to reduce
        
        # Compress the dynamic range
        new_min = l_mean - (l_range * compression_factor / 2.0)
        new_max = l_mean + (l_range * compression_factor / 2.0)
        
        # Clamp and scale L channel
        l_compressed = np.clip(l, new_min, new_max)
        l_compressed = ((l_compressed - new_min) / (new_max - new_min + 1e-6)) * 255.0
        
        # Blend with original for smoother result
        l_final = l_compressed * blend_ratio + l * (1 - blend_ratio)
        
        # Merge channels and convert back to RGB
        lab_reduced = np.stack([l_final, a, b], axis=2)
        lab_reduced = np.clip(lab_reduced, 0, 255).astype(np.uint8)
        reduced = cv2.cvtColor(lab_reduced, cv2.COLOR_LAB2RGB)
        
        return reduced
    
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
        Uses adaptive parameters based on image contrast level.
        
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
            
            # Calculate current contrast (standard deviation of L channel)
            l_std = np.std(l.astype(np.float32))
            
            # Adaptive CLAHE parameters based on contrast level
            if l_std < 15:  # Very low contrast - use stronger enhancement
                clip_limit = 3.0
                tile_size = (4, 4)  # Smaller tiles for more aggressive enhancement
            elif l_std < 25:  # Low contrast
                clip_limit = 2.5
                tile_size = (6, 6)
            else:  # Moderate contrast
                clip_limit = 2.0
                tile_size = (8, 8)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
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
    
    def fix_color_shift(self, image: np.ndarray) -> np.ndarray:
        """
        Fix overall color shift by balancing RGB channels using white balance.
        Uses a very conservative approach to avoid over-correction.
        DISABLED: This method can cause blue tint issues. Returning original image.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Color-corrected image (currently returns original to avoid blue tint)
        """
        # TEMPORARILY DISABLED: Color shift fix is causing blue tint issues
        # Return original image to prevent color distortion
        return image.copy()
        
        # Original implementation (disabled):
        # # Convert to float for calculations
        # img_float = image.astype(np.float32)
        # 
        # # Compute mean values for each channel
        # r_mean = np.mean(img_float[:, :, 0])
        # g_mean = np.mean(img_float[:, :, 1])
        # b_mean = np.mean(img_float[:, :, 2])
        # 
        # # Use gray world assumption: average of all channels should be similar
        # # Target: make all channel means equal to the average of the three
        # target_mean = (r_mean + g_mean + b_mean) / 3.0
        # 
        # # Only correct if there's a significant imbalance (avoid over-correction)
        # max_deviation = max(abs(r_mean - target_mean), 
        #                    abs(g_mean - target_mean), 
        #                    abs(b_mean - target_mean))
        # 
        # # Only apply correction if deviation is significant (> 20) and very conservative
        # if max_deviation < 20.0:
        #     return image  # No significant color shift, return original
        # 
        # # Compute adjustment factors (extremely conservative - only 15% correction)
        # correction_strength = 0.15
        # if r_mean > 0:
        #     r_factor = 1.0 + (target_mean / r_mean - 1.0) * correction_strength
        #     r_factor = np.clip(r_factor, 0.85, 1.15)  # Limit to ±15% change
        # else:
        #     r_factor = 1.0
        # if g_mean > 0:
        #     g_factor = 1.0 + (target_mean / g_mean - 1.0) * correction_strength
        #     g_factor = np.clip(g_factor, 0.85, 1.15)  # Limit to ±15% change
        # else:
        #     g_factor = 1.0
        # if b_mean > 0:
        #     b_factor = 1.0 + (target_mean / b_mean - 1.0) * correction_strength
        #     b_factor = np.clip(b_factor, 0.85, 1.15)  # Limit to ±15% change
        # else:
        #     b_factor = 1.0
        # 
        # # Apply correction
        # corrected = img_float.copy()
        # corrected[:, :, 0] = corrected[:, :, 0] * r_factor
        # corrected[:, :, 1] = corrected[:, :, 1] * g_factor
        # corrected[:, :, 2] = corrected[:, :, 2] * b_factor
        # 
        # # Clip to valid range
        # corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        # 
        # return corrected
    
    def fix_color_diffusion(self, image: np.ndarray) -> np.ndarray:
        """
        Fix color diffusion by sharpening a and b channels in Lab space.
        Made more conservative to avoid color distortion.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Color-diffusion-corrected image
        """
        # Convert to Lab color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Apply very gentle unsharp mask to a and b channels
        # Create very mild sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * 0.05  # Very gentle sharpening (reduced from 0.1)
        
        a_sharpened = cv2.filter2D(a_channel, -1, kernel)
        b_sharpened = cv2.filter2D(b_channel, -1, kernel)
        
        # Blend with original (75% original, 25% sharpened - very conservative)
        a_corrected = a_channel * 0.75 + a_sharpened * 0.25
        b_corrected = b_channel * 0.75 + b_sharpened * 0.25
        
        # Merge channels
        lab_corrected = np.stack([l_channel, a_corrected, b_corrected], axis=2)
        lab_corrected = np.clip(lab_corrected, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)
        
        return corrected
    
    def fix_green_channel_shift(self, image: np.ndarray) -> np.ndarray:
        """
        Fix green channel shift by correcting green channel based on R and B channels.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Green-channel-corrected image
        """
        # Extract channels
        r_channel = image[:, :, 0].astype(np.float32)
        g_channel = image[:, :, 1].astype(np.float32)
        b_channel = image[:, :, 2].astype(np.float32)
        
        # Compute expected green channel (average of R and B)
        expected_g = (r_channel + b_channel) / 2.0
        
        # Compute gradient magnitude mask (normalized)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)
        
        # Normalize gradient magnitude to [0, 1]
        if gradient_magnitude.max() > 0:
            gradient_mask = gradient_magnitude / gradient_magnitude.max()
        else:
            gradient_mask = np.zeros_like(gradient_magnitude)
        
        # Invert mask (apply correction more in non-edge regions)
        correction_mask = 1.0 - gradient_mask * 0.5  # Gentle correction
        
        # Correct green channel by blending with expected
        g_corrected = g_channel * correction_mask + expected_g * (1.0 - correction_mask)
        
        # Merge channels
        corrected = np.stack([r_channel, g_corrected, b_channel], axis=2)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
    
    def adjust_saturation(self, image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Adjust image saturation with proper handling and validation.
        
        Args:
            image: RGB image array (H, W, 3)
            factor: Saturation factor (1.0 = no change, >1.0 = increase, <1.0 = decrease)
            
        Returns:
            Saturation-adjusted image
        """
        if factor == 1.0:
            return image  # No change needed
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Store original saturation for validation
        original_sat = hsv[:, :, 1].copy()
        
        # Adjust saturation channel
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        
        # Clip to valid range [0, 255]
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Ensure H and V channels are also in valid range
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)  # Hue is 0-179 in OpenCV
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)  # Value
        
        # Convert back to RGB
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Verify the adjustment was applied (check if saturation actually changed)
        hsv_check = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV).astype(np.float32)
        mean_sat_before = np.mean(original_sat)
        mean_sat_after = np.mean(hsv_check[:, :, 1])
        
        # If adjustment didn't work (e.g., image was already at limits), try alternative method
        if abs(mean_sat_after - mean_sat_before) < 1.0 and abs(factor - 1.0) > 0.1:
            # Use LAB color space as alternative for saturation adjustment
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)
            
            # Adjust a and b channels (color channels) to change saturation
            if factor > 1.0:  # Increase saturation
                a = a * (1.0 + (factor - 1.0) * 0.5)  # More conservative increase
                b = b * (1.0 + (factor - 1.0) * 0.5)
            else:  # Decrease saturation
                a = a * factor
                b = b * factor
            
            # Clip a and b channels (valid range is roughly -128 to 127, but we'll use 0-255)
            a = np.clip(a, 0, 255)
            b = np.clip(b, 0, 255)
            
            # Merge and convert back
            lab_adjusted = np.stack([l, a, b], axis=2)
            lab_adjusted = np.clip(lab_adjusted, 0, 255).astype(np.uint8)
            adjusted = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2RGB)
        
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
    
    def preprocess_image_with_verification(self, image: np.ndarray,
                                          target_size: Tuple[int, int] = (224, 224),
                                          blur_threshold: float = 100.0,
                                          sharpness_threshold: float = 500.0,
                                          noise_threshold: float = 20.0,
                                          dark_threshold: float = 50.0,
                                          bright_threshold: float = 200.0,
                                          target_mean: float = 128.0,
                                          low_contrast_threshold: float = 30.0,
                                          high_contrast_threshold: float = 80.0,
                                          max_iterations: int = 3) -> Tuple[np.ndarray, Dict]:
        """
        Comprehensive preprocessing with verification loop.
        Re-measures metrics after each fix and escalates if needed.
        
        Returns:
            Tuple of (processed_image, fix_log)
        """
        fix_log = {
            'fixes_applied': [],
            'iterations': 0,
            'final_metrics': {}
        }
        
        # Resize first
        processed = self.resize_image(image, target_size=target_size)
        
        # Verification loop
        for iteration in range(max_iterations):
            fix_log['iterations'] = iteration + 1
            
            # Compute current metrics
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            
            # Blur/Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = laplacian.var()
            
            # Brightness
            brightness = np.mean(gray)
            
            # Contrast (std)
            contrast = np.std(gray)
            
            # Noise (gradient std)
            h_diff = np.diff(gray.astype(np.float64), axis=1)
            v_diff = np.diff(gray.astype(np.float64), axis=0)
            gradients = np.concatenate([h_diff.flatten(), v_diff.flatten()])
            noise_score = np.std(gradients)
            
            # Check and apply fixes
            needs_fix = False
            
            # 1. Check sharpness (over-sharpened)
            if blur_score > sharpness_threshold:
                processed = self.normalize_sharpness(processed)
                fix_log['fixes_applied'].append(f'iteration_{iteration+1}: normalize_sharpness')
                needs_fix = True
            
            # 2. Check noise
            if noise_score > noise_threshold:
                processed = self.denoise_image(processed, method='fastNlMeans')
                fix_log['fixes_applied'].append(f'iteration_{iteration+1}: denoise')
                needs_fix = True
            
            # 3. Check brightness
            if brightness < dark_threshold or brightness > bright_threshold:
                processed = self.normalize_brightness(processed, target_brightness=target_mean)
                fix_log['fixes_applied'].append(f'iteration_{iteration+1}: normalize_brightness')
                needs_fix = True
            
            # 4. Check contrast
            if contrast < low_contrast_threshold:
                processed = self.enhance_contrast(processed, method='CLAHE')
                fix_log['fixes_applied'].append(f'iteration_{iteration+1}: enhance_contrast')
                needs_fix = True
            elif contrast > high_contrast_threshold:
                processed = self.reduce_contrast(processed)
                fix_log['fixes_applied'].append(f'iteration_{iteration+1}: reduce_contrast')
                needs_fix = True
            
            # If no fixes needed, break
            if not needs_fix:
                break
        
        # Final metrics
        gray_final = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        laplacian_final = cv2.Laplacian(gray_final, cv2.CV_64F)
        h_diff_final = np.diff(gray_final.astype(np.float64), axis=1)
        v_diff_final = np.diff(gray_final.astype(np.float64), axis=0)
        gradients_final = np.concatenate([h_diff_final.flatten(), v_diff_final.flatten()])
        
        fix_log['final_metrics'] = {
            'blur_score': laplacian_final.var(),
            'brightness': np.mean(gray_final),
            'contrast': np.std(gray_final),
            'noise': np.std(gradients_final)
        }
        
        return processed, fix_log
    
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

    def _get_output_class_images(self) -> Dict[str, List[Path]]:
        """Collect saved image paths per class from the cleaned dataset output."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        class_images = {}

        for class_dir in sorted(self.output_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            images = [
                path for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in valid_extensions
            ]
            if images:
                class_images[class_dir.name] = images

        return class_images

    def balance_dataset(self, strategy: str = 'hybrid', target_count: int = None) -> Dict:
        """
        Balance the cleaned dataset after quality fixes have been applied.

        Strategies:
        - undersample: reduce all classes to the minority count
        - oversample: augment all classes to the majority count
        - hybrid: undersample majorities and augment minorities toward the median count
        """
        rng = random.Random(42)
        class_images = self._get_output_class_images()

        if len(class_images) < 2:
            return {
                'applied': False,
                'strategy': strategy,
                'target_count': 0,
                'before_distribution': {k: len(v) for k, v in class_images.items()},
                'after_distribution': {k: len(v) for k, v in class_images.items()},
                'removed': 0,
                'augmented': 0,
            }

        before_distribution = {class_name: len(images) for class_name, images in class_images.items()}
        counts = list(before_distribution.values())

        if target_count is None:
            if strategy == 'undersample':
                target_count = min(counts)
            elif strategy == 'oversample':
                target_count = max(counts)
            else:
                target_count = max(1, int(round(float(np.median(counts)))))

        removed_count = 0
        augmented_count = 0

        # Reduce majority classes first.
        if strategy in {'undersample', 'hybrid'}:
            for class_name, images in class_images.items():
                if len(images) <= target_count:
                    continue

                to_remove = rng.sample(images, len(images) - target_count)
                for image_path in to_remove:
                    image_path.unlink(missing_ok=True)
                    removed_count += 1

        class_images = self._get_output_class_images()

        # Then grow minority classes if the chosen strategy requires it.
        if strategy in {'oversample', 'hybrid'}:
            for class_name, images in list(class_images.items()):
                current_count = len(images)
                if current_count == 0 or current_count >= target_count:
                    continue

                for aug_index in range(target_count - current_count):
                    source_path = rng.choice(images)
                    image = self.data_loader.load_image(str(source_path))
                    augmented = self.augment_image(image, augmentation_type='random')
                    augmented = self.resize_image(augmented, target_size=(224, 224))

                    aug_path = source_path.parent / f"{source_path.stem}_balanced_{aug_index}{source_path.suffix}"
                    while aug_path.exists():
                        aug_index += 1
                        aug_path = source_path.parent / f"{source_path.stem}_balanced_{aug_index}{source_path.suffix}"

                    augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_path), augmented_bgr)
                    augmented_count += 1

        after_distribution = {
            class_name: len(images)
            for class_name, images in self._get_output_class_images().items()
        }

        return {
            'applied': True,
            'strategy': strategy,
            'target_count': target_count,
            'before_distribution': before_distribution,
            'after_distribution': after_distribution,
            'removed': removed_count,
            'augmented': augmented_count,
        }

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
        balance_action = None
        
        # Track which images need which fixes
        image_fixes = {}  # {image_index: [list of fixes to apply]}
        
        for action in actions:
            action_type = action['type']
            if action_type == 'balance_dataset':
                balance_action = action
                continue

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
                elif action_type == 'contrast_enhance':
                    image_fixes[idx].append({
                        'type': 'contrast_enhance',
                        'method': action.get('method', 'CLAHE'),
                        'reason': reason
                    })
                elif action_type == 'contrast_reduce':
                    image_fixes[idx].append({
                        'type': 'contrast_reduce',
                        'reason': reason
                    })
                elif action_type == 'normalize_sharpness':
                    image_fixes[idx].append({
                        'type': 'normalize_sharpness',
                        'reason': reason
                    })
                elif action_type == 'adjust_saturation':
                    image_fixes[idx].append({
                        'type': 'adjust_saturation',
                        'saturation_factor': action.get('saturation_factor', 1.0),
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
                
                # Track all fixes to apply
                apply_denoising = False
                denoise_method = 'fastNlMeans'
                apply_brightness = False
                target_brightness = 128.0
                apply_contrast_enhance = False
                contrast_method = 'CLAHE'
                apply_contrast_reduce = False
                apply_sharpness_normalize = False
                apply_saturation_adjust = False
                saturation_factor = 1.0
                
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
                    elif fix['type'] == 'contrast_enhance':
                        apply_contrast_enhance = True
                        contrast_method = fix.get('method', 'CLAHE')
                        fix_descriptions.append(f"contrast_enhance({contrast_method})")
                    elif fix['type'] == 'contrast_reduce':
                        apply_contrast_reduce = True
                        fix_descriptions.append("contrast_reduce")
                    elif fix['type'] == 'normalize_sharpness':
                        apply_sharpness_normalize = True
                        fix_descriptions.append("normalize_sharpness")
                    elif fix['type'] == 'adjust_saturation':
                        apply_saturation_adjust = True
                        saturation_factor = fix.get('saturation_factor', 1.0)
                        fix_descriptions.append(f"adjust_saturation({saturation_factor:.2f})")
                
                # Load image
                image = self.data_loader.load_image(img_path)
                if image is None:
                    skipped_count += 1
                    continue
                
                # Apply fixes in correct order
                # 0. Resize first (all images must be consistent size for model)
                image = self.resize_image(image, target_size=(224, 224))
                fix_descriptions.append("resize(224x224)")
                
                # 1. Denoising first (removes noise before other enhancements)
                if apply_denoising:
                    image = self.denoise_image(image, method=denoise_method)
                
                # 2. Brightness normalization
                if apply_brightness:
                    image = self.normalize_brightness(image, target_brightness=target_brightness)
                
                # 3. Contrast enhancement (if low contrast)
                if apply_contrast_enhance:
                    image = self.enhance_contrast(image, method=contrast_method)
                
                # 4. Contrast reduction (if high contrast)
                if apply_contrast_reduce:
                    image = self.reduce_contrast(image)
                
                # 5. Saturation adjustment
                if apply_saturation_adjust:
                    image = self.adjust_saturation(image, factor=saturation_factor)
                
                # 6. Sharpness normalization (last, as it affects overall image)
                if apply_sharpness_normalize:
                    image = self.normalize_sharpness(image)
                
                fixed_image = image
                
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
                
                # Load and resize image (all images must be consistent size for model)
                image = self.data_loader.load_image(img_path)
                if image is None:
                    skipped_count += 1
                    continue
                
                # Resize to consistent size
                image = self.resize_image(image, target_size=(224, 224))
                
                img_path_obj = Path(img_path)
                relative_path = img_path_obj.relative_to(self.data_loader.dataset_path)
                output_path = self.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save resized image
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), image_bgr)
                skipped_count += 1  # Count as "skipped" (no fix needed, just resized)
            except Exception as e:
                print(f"  Warning: Could not process {image_paths[idx]}: {e}")
                continue
        
        excluded_count = len(excluded_indices)
        if balance_action is not None:
            print("Applying dataset balancing...")
            balance_summary = self.balance_dataset(
                strategy=balance_action.get('strategy', 'hybrid'),
                target_count=balance_action.get('target_count')
            )
            if balance_summary.get('applied'):
                applied_fixes_log.append(
                    "Balanced dataset "
                    f"using {balance_summary['strategy']} strategy to target "
                    f"{balance_summary['target_count']} images per class "
                    f"(removed {balance_summary['removed']}, augmented {balance_summary['augmented']})"
                )
                print(
                    "  Balanced dataset to target "
                    f"{balance_summary['target_count']} images per class "
                    f"using {balance_summary['strategy']}"
                )

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
