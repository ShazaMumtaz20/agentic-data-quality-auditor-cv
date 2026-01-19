"""
Quality Metrics Module
Computes various quality metrics for images: blur, noise, brightness, etc.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from data_loader import DataLoader


class QualityMetrics:
    """
    Computes quality metrics for images in a dataset.
    """
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize quality metrics calculator.
        
        Args:
            data_loader: DataLoader instance with loaded dataset
        """
        self.data_loader = data_loader
        self.blur_scores = []
        self.noise_scores = []
        self.brightness_scores = []
        self.contrast_scores = []
        self.saturation_scores = []
        self.edge_density_scores = []
        self.color_shift_scores = []
        self.color_diffusion_scores = []
        self.green_channel_shift_scores = []
        self.corruption_flags = []
        self.image_paths = []
        
    def compute_blur_score(self, image: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.
        Higher values indicate sharper images.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Blur score (Laplacian variance)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Return variance as blur score
        return laplacian.var()
    
    def compute_noise_score(self, image: np.ndarray) -> float:
        """
        Estimate noise level using standard deviation of pixel differences.
        Higher values indicate more noise.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Noise score (standard deviation of local differences)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
        
        # Compute horizontal and vertical gradients
        h_diff = np.diff(gray, axis=1)
        v_diff = np.diff(gray, axis=0)
        
        # Combine gradients and compute standard deviation
        gradients = np.concatenate([h_diff.flatten(), v_diff.flatten()])
        noise_score = np.std(gradients)
        
        return noise_score
    
    def compute_brightness(self, image: np.ndarray) -> float:
        """
        Compute mean brightness of the image.
        Returns value between 0 (dark) and 255 (bright).
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Mean brightness value
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Return mean pixel intensity
        return np.mean(gray)
    
    def compute_contrast_score(self, image: np.ndarray) -> float:
        """
        Compute contrast score using standard deviation of pixel intensities.
        Higher values indicate better contrast.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Contrast score (standard deviation)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Return standard deviation as contrast measure
        return np.std(gray)
    
    def detect_color_shift(self, image: np.ndarray) -> float:
        """
        Detect overall color shift (e.g., too red, too blue, etc.).
        Measures the deviation of mean channel values from expected balance.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Color shift score (higher = more color shift)
        """
        # Compute mean values for each channel
        r_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        b_mean = np.mean(image[:, :, 2])
        
        # Expected balanced values (assuming balanced image has similar means)
        # Or compute expected based on grayscale mean
        gray_mean = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        
        # Deviation from expected balance
        r_dev = abs(r_mean - gray_mean)
        g_dev = abs(g_mean - gray_mean)
        b_dev = abs(b_mean - gray_mean)
        
        # Color shift score is the maximum deviation
        color_shift = max(r_dev, g_dev, b_dev)
        
        return color_shift
    
    def detect_color_diffusion(self, image: np.ndarray) -> float:
        """
        Detect color diffusion where a and b channels in Lab space have been Gaussian blurred.
        Measures the smoothness/variance of a and b channels.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Color diffusion score (lower = more diffusion/blur in color channels)
        """
        # Convert to Lab color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        a_channel = lab[:, :, 1].astype(np.float64)
        b_channel = lab[:, :, 2].astype(np.float64)
        
        # Compute Laplacian variance for a and b channels (similar to blur detection)
        laplacian_a = cv2.Laplacian(a_channel, cv2.CV_64F)
        laplacian_b = cv2.Laplacian(b_channel, cv2.CV_64F)
        
        # Lower variance indicates more blur/diffusion in color channels
        variance_a = laplacian_a.var()
        variance_b = laplacian_b.var()
        
        # Average variance (lower = more color diffusion)
        color_diffusion_score = (variance_a + variance_b) / 2.0
        
        return color_diffusion_score
    
    def detect_green_channel_shift(self, image: np.ndarray) -> float:
        """
        Detect color shift where green channel has been randomly translated and blended.
        Measures correlation between green channel and gradient magnitude mask.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Green channel shift score (higher = more likely to have green channel shift)
        """
        # Extract channels
        r_channel = image[:, :, 0].astype(np.float64)
        g_channel = image[:, :, 1].astype(np.float64)
        b_channel = image[:, :, 2].astype(np.float64)
        
        # Compute gradient magnitude (normalized) as mask
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient magnitude to [0, 1]
        if gradient_magnitude.max() > 0:
            gradient_mask = gradient_magnitude / gradient_magnitude.max()
        else:
            gradient_mask = np.zeros_like(gradient_magnitude)
        
        # Compute expected green channel (average of R and B)
        expected_g = (r_channel + b_channel) / 2.0
        
        # Measure deviation of green channel from expected
        g_deviation = np.abs(g_channel - expected_g)
        
        # Weight by gradient mask (higher weight in edge regions)
        weighted_deviation = g_deviation * gradient_mask
        
        # Score is the mean weighted deviation
        green_shift_score = np.mean(weighted_deviation)
        
        return green_shift_score
    
    def compute_saturation_score(self, image: np.ndarray) -> float:
        """
        Compute saturation score. Higher values indicate over-saturation.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Saturation score (mean saturation)
        """
        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract saturation channel
        saturation = hsv[:, :, 1]
        
        # Return mean saturation
        return np.mean(saturation)
    
    def compute_edge_density(self, image: np.ndarray) -> float:
        """
        Compute edge density using gradient magnitude.
        Higher values indicate more edges (sharper image).
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Edge density score
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return mean gradient magnitude as edge density
        return np.mean(magnitude)
    
    def detect_corruption(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Detect if image is corrupted (NaNs, extreme values, etc.).
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Tuple of (is_corrupted: bool, reason: str)
        """
        # Check for NaN or Inf values
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            return True, "Contains NaN or Inf values"
        
        # Check for extreme pixel values (outside valid range)
        if np.any(image < 0) or np.any(image > 255):
            return True, "Contains out-of-range pixel values"
        
        # Check if image is too small
        if image.shape[0] < 10 or image.shape[1] < 10:
            return True, "Image too small"
        
        # Check if image is all zeros or all same value (likely corrupted)
        if np.std(image) < 1.0:
            return True, "Image has no variation (likely corrupted)"
        
        return False, "OK"
    
    def analyze_dataset(self) -> Dict[str, List[float]]:
        """
        Analyze all images in the dataset and compute quality metrics.
        
        Returns:
            Dictionary containing lists of metrics for all images
        """
        dataset_dict = self.data_loader.load_dataset()
        
        self.blur_scores = []
        self.noise_scores = []
        self.brightness_scores = []
        self.contrast_scores = []
        self.saturation_scores = []
        self.edge_density_scores = []
        self.color_shift_scores = []
        self.color_diffusion_scores = []
        self.green_channel_shift_scores = []
        self.corruption_flags = []
        self.image_paths = []
        
        print("Analyzing dataset quality...")
        total_images = sum(len(paths) for paths in dataset_dict.values())
        processed = 0
        
        for class_name, image_paths in dataset_dict.items():
            for img_path in image_paths:
                try:
                    # Load image
                    image = self.data_loader.load_image(img_path)
                    
                    # Check for corruption first
                    is_corrupted, corruption_reason = self.detect_corruption(image)
                    
                    if is_corrupted:
                        # Mark as corrupted but still store metrics (will be excluded later)
                        self.corruption_flags.append(True)
                        self.blur_scores.append(0.0)
                        self.noise_scores.append(0.0)
                        self.brightness_scores.append(0.0)
                        self.contrast_scores.append(0.0)
                        self.saturation_scores.append(0.0)
                        self.edge_density_scores.append(0.0)
                        self.color_shift_scores.append(0.0)
                        self.color_diffusion_scores.append(0.0)
                        self.green_channel_shift_scores.append(0.0)
                        self.image_paths.append(img_path)
                        processed += 1
                        if processed % 10 == 0:
                            print(f"  Processed {processed}/{total_images} images...")
                        continue
                    
                    # Compute all metrics
                    blur = self.compute_blur_score(image)
                    noise = self.compute_noise_score(image)
                    brightness = self.compute_brightness(image)
                    contrast = self.compute_contrast_score(image)
                    saturation = self.compute_saturation_score(image)
                    edge_density = self.compute_edge_density(image)
                    color_shift = self.detect_color_shift(image)
                    color_diffusion = self.detect_color_diffusion(image)
                    green_shift = self.detect_green_channel_shift(image)
                    
                    # Store results
                    self.blur_scores.append(blur)
                    self.noise_scores.append(noise)
                    self.brightness_scores.append(brightness)
                    self.contrast_scores.append(contrast)
                    self.saturation_scores.append(saturation)
                    self.edge_density_scores.append(edge_density)
                    self.color_shift_scores.append(color_shift)
                    self.color_diffusion_scores.append(color_diffusion)
                    self.green_channel_shift_scores.append(green_shift)
                    self.corruption_flags.append(False)
                    self.image_paths.append(img_path)
                    
                    processed += 1
                    if processed % 10 == 0:
                        print(f"  Processed {processed}/{total_images} images...")
                        
                except Exception as e:
                    # Mark as corrupted if loading/processing fails
                    print(f"  Warning: Could not process {img_path}: {e}")
                    self.corruption_flags.append(True)
                    self.blur_scores.append(0.0)
                    self.noise_scores.append(0.0)
                    self.brightness_scores.append(0.0)
                    self.contrast_scores.append(0.0)
                    self.saturation_scores.append(0.0)
                    self.edge_density_scores.append(0.0)
                    self.image_paths.append(img_path)
                    processed += 1
                    continue
        
        print(f"Analysis complete! Processed {processed} images.")
        
        return {
            'blur_scores': self.blur_scores,
            'noise_scores': self.noise_scores,
            'brightness_scores': self.brightness_scores,
            'contrast_scores': self.contrast_scores,
            'saturation_scores': self.saturation_scores,
            'edge_density_scores': self.edge_density_scores,
            'color_shift_scores': self.color_shift_scores,
            'color_diffusion_scores': self.color_diffusion_scores,
            'green_channel_shift_scores': self.green_channel_shift_scores,
            'corruption_flags': self.corruption_flags,
            'image_paths': self.image_paths
        }
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of all metrics.
        
        Returns:
            Dictionary with statistics for each metric
        """
        stats = {}
        
        if len(self.blur_scores) > 0:
            stats['blur'] = {
                'mean': np.mean(self.blur_scores),
                'std': np.std(self.blur_scores),
                'min': np.min(self.blur_scores),
                'max': np.max(self.blur_scores),
                'median': np.median(self.blur_scores)
            }
        
        if len(self.noise_scores) > 0:
            stats['noise'] = {
                'mean': np.mean(self.noise_scores),
                'std': np.std(self.noise_scores),
                'min': np.min(self.noise_scores),
                'max': np.max(self.noise_scores),
                'median': np.median(self.noise_scores)
            }
        
        if len(self.brightness_scores) > 0:
            stats['brightness'] = {
                'mean': np.mean(self.brightness_scores),
                'std': np.std(self.brightness_scores),
                'min': np.min(self.brightness_scores),
                'max': np.max(self.brightness_scores),
                'median': np.median(self.brightness_scores)
            }
        
        if len(self.contrast_scores) > 0:
            stats['contrast'] = {
                'mean': np.mean(self.contrast_scores),
                'std': np.std(self.contrast_scores),
                'min': np.min(self.contrast_scores),
                'max': np.max(self.contrast_scores),
                'median': np.median(self.contrast_scores)
            }
        
        if len(self.saturation_scores) > 0:
            stats['saturation'] = {
                'mean': np.mean(self.saturation_scores),
                'std': np.std(self.saturation_scores),
                'min': np.min(self.saturation_scores),
                'max': np.max(self.saturation_scores),
                'median': np.median(self.saturation_scores)
            }
        
        if len(self.edge_density_scores) > 0:
            stats['edge_density'] = {
                'mean': np.mean(self.edge_density_scores),
                'std': np.std(self.edge_density_scores),
                'min': np.min(self.edge_density_scores),
                'max': np.max(self.edge_density_scores),
                'median': np.median(self.edge_density_scores)
            }
        
        # Sharpness (same as blur but for over-sharpening detection)
        if len(self.blur_scores) > 0:
            stats['sharpness'] = {
                'mean': np.mean(self.blur_scores),
                'std': np.std(self.blur_scores),
                'min': np.min(self.blur_scores),
                'max': np.max(self.blur_scores),
                'median': np.median(self.blur_scores)
            }
        
        if len(self.corruption_flags) > 0:
            corrupted_count = sum(self.corruption_flags)
            stats['corruption'] = {
                'corrupted_count': corrupted_count,
                'corrupted_percentage': (corrupted_count / len(self.corruption_flags)) * 100
            }
        
        return stats
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get class distribution from the data loader.
        
        Returns:
            Dictionary mapping class names to image counts
        """
        return self.data_loader.get_class_distribution()
