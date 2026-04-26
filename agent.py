"""
Agentic Decision Engine Module
Makes intelligent decisions about dataset quality issues and decides on concrete actions.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np


class QualityAgent:
    """
    Rule-based agent that analyzes dataset statistics and decides on concrete actions.
    """
    
    def __init__(self):
        """Initialize the quality agent with default thresholds."""
        # Thresholds for quality assessment
        self.BLUR_THRESHOLD = 100.0  # Laplacian variance below this = blurry (REMOVE)
        self.SHARPNESS_THRESHOLD = 500.0  # Laplacian variance above this = over-sharpened
        self.NOISE_THRESHOLD = 20.0  # Noise score above this = noisy
        self.DARK_THRESHOLD = 50.0  # Mean brightness below this = too dark
        self.BRIGHT_THRESHOLD = 200.0  # Mean brightness above this = too bright
        self.TARGET_MEAN = 128.0  # Target brightness
        self.LOW_CONTRAST_THRESHOLD = 30.0  # Std below this = low contrast
        self.HIGH_CONTRAST_THRESHOLD = 80.0  # Std above this = high contrast
        self.CLASS_IMBALANCE_RATIO = 0.3  # If one class has <30% of average, it's imbalanced
        self.MAX_CLASS_REMOVAL_RATIO = 0.4  # Don't remove more than 40% from any class
        self.LOW_SATURATION_THRESHOLD = 30.0  # Saturation below this = under-saturated
        self.HIGH_SATURATION_THRESHOLD = 200.0  # Saturation above this = over-saturated
        self.BALANCE_TOLERANCE = 0  # Classes must match exactly after balancing
        
    def analyze(self, metrics_stats: Dict, class_distribution: Dict[str, int]) -> Dict:
        """
        Analyze dataset quality and generate recommendations.
        
        Args:
            metrics_stats: Statistics from QualityMetrics.get_statistics()
            class_distribution: Dictionary mapping class names to counts
            
        Returns:
            Dictionary containing detected issues and recommendations
        """
        issues = []
        recommendations = []
        decisions = []
        
        # Analyze blur
        if 'blur' in metrics_stats:
            blur_stats = metrics_stats['blur']
            blur_mean = blur_stats['mean']
            
            # Count blurry images (assuming we have access to individual scores)
            # For now, use mean as indicator
            if blur_mean < self.BLUR_THRESHOLD:
                blur_percentage = self._estimate_blur_percentage(blur_mean, self.BLUR_THRESHOLD)
                issues.append({
                    'type': 'blur',
                    'severity': 'high' if blur_mean < self.BLUR_THRESHOLD * 0.5 else 'medium',
                    'description': f'Dataset has blur issues (mean blur score: {blur_mean:.2f})'
                })
                decisions.append(f"{blur_percentage:.0f}% images are blurry → recommend denoising")
                recommendations.append("Apply denoising filter to blurry images")
        
        # Analyze noise
        if 'noise' in metrics_stats:
            noise_stats = metrics_stats['noise']
            noise_mean = noise_stats['mean']
            
            if noise_mean > self.NOISE_THRESHOLD:
                noise_percentage = self._estimate_noise_percentage(noise_mean, self.NOISE_THRESHOLD)
                issues.append({
                    'type': 'noise',
                    'severity': 'high' if noise_mean > self.NOISE_THRESHOLD * 1.5 else 'medium',
                    'description': f'Dataset has high noise levels (mean noise score: {noise_mean:.2f})'
                })
                decisions.append(f"{noise_percentage:.0f}% images are noisy → recommend denoising")
                recommendations.append("Apply noise reduction filter")
        
        # Analyze brightness
        if 'brightness' in metrics_stats:
            brightness_stats = metrics_stats['brightness']
            brightness_mean = brightness_stats['mean']
            
            if brightness_mean < self.DARK_THRESHOLD:
                issues.append({
                    'type': 'brightness',
                    'severity': 'medium',
                    'description': f'Dataset is too dark (mean brightness: {brightness_mean:.2f})'
                })
                decisions.append("Dataset too dark → recommend brightness normalization")
                recommendations.append("Apply brightness normalization to improve visibility")
            elif brightness_mean > self.BRIGHT_THRESHOLD:
                issues.append({
                    'type': 'brightness',
                    'severity': 'medium',
                    'description': f'Dataset is too bright (mean brightness: {brightness_mean:.2f})'
                })
                decisions.append("Dataset too bright → recommend brightness normalization")
                recommendations.append("Apply brightness normalization to reduce overexposure")
        
        # Analyze class distribution
        if len(class_distribution) > 0:
            class_counts = list(class_distribution.values())
            total_images = sum(class_counts)
            avg_count = total_images / len(class_distribution)
            
            imbalanced_classes = []
            for class_name, count in class_distribution.items():
                ratio = count / avg_count if avg_count > 0 else 0
                if ratio < self.CLASS_IMBALANCE_RATIO:
                    imbalanced_classes.append((class_name, count, ratio))
            
            if imbalanced_classes:
                for class_name, count, ratio in imbalanced_classes:
                    issues.append({
                        'type': 'class_imbalance',
                        'severity': 'high',
                        'description': f'Class "{class_name}" is under-represented ({count} images, {ratio*100:.1f}% of average)'
                    })
                    decisions.append(f'Class "{class_name}" is under-represented → recommend oversampling')
                    recommendations.append(f"Apply oversampling or data augmentation for class '{class_name}'")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'decisions': decisions,
            'summary': self._generate_summary(issues, decisions)
        }
    
    def _estimate_blur_percentage(self, mean_blur: float, threshold: float) -> float:
        """
        Estimate percentage of blurry images based on mean blur score.
        This is a heuristic estimation.
        """
        if mean_blur >= threshold:
            return 0.0
        # Linear interpolation: if mean is 0, assume 100% blurry
        # If mean is at threshold, assume 0% blurry
        percentage = (1 - mean_blur / threshold) * 100
        return max(0, min(100, percentage))
    
    def _estimate_noise_percentage(self, mean_noise: float, threshold: float) -> float:
        """
        Estimate percentage of noisy images based on mean noise score.
        This is a heuristic estimation.
        """
        if mean_noise <= threshold:
            return 0.0
        # Linear interpolation: if mean is 2x threshold, assume 100% noisy
        # If mean is at threshold, assume 0% noisy
        excess = mean_noise - threshold
        percentage = (excess / threshold) * 50  # Scale factor
        return max(0, min(100, percentage))
    
    def _generate_summary(self, issues: List[Dict], decisions: List[str], excluded_count: int = 0) -> str:
        """
        Generate a human-readable summary of the analysis.
        """
        if len(issues) == 0:
            return "✓ Dataset quality is good. No major issues detected."
        
        summary = f"Detected {len(issues)} issue(s):\n"
        for i, issue in enumerate(issues, 1):
            summary += f"  {i}. [{issue['severity'].upper()}] {issue['description']}\n"
        
        if excluded_count > 0:
            summary += f"\n⚠ {excluded_count} images will be EXCLUDED from cleaned dataset (irreversible defects)"
        
        return summary
    
    def decide_actions(self, blur_scores: List[float], 
                      noise_scores: List[float],
                      brightness_scores: List[float],
                      image_paths: List[str],
                      class_distribution: Dict[str, int],
                      contrast_scores: Optional[List[float]] = None,
                      saturation_scores: Optional[List[float]] = None,
                      corruption_flags: Optional[List[bool]] = None,
                      auto_balance: bool = False,
                      balance_strategy: str = 'hybrid') -> Dict:
        """
        Analyze dataset and decide on concrete actions following realistic ML engineering rules.
        Blur is irreversible - exclude severely blurred images instead of trying to fix.
        
        Supported fixes: blur removal (exclusion), noise reduction, brightness normalization,
        contrast adjustment, sharpness normalization, and saturation adjustment.
        All images are resized to 224x224 for model consistency.
        
        Args:
            blur_scores: List of blur scores for each image
            noise_scores: List of noise scores for each image
            brightness_scores: List of brightness scores for each image
            image_paths: List of paths to images
            class_distribution: Dictionary mapping class names to counts
            contrast_scores: Optional list of contrast scores
            saturation_scores: Optional list of saturation scores
            corruption_flags: Optional list of corruption flags
            
        Returns:
            Dictionary containing:
            - issues: Detected problems
            - actions: List of concrete actions to take (for fixer)
            - action_plan: Human-readable action plan
            - per_image_actions: Detailed per-image decisions
            - excluded_images: List of excluded images with reasons
            - summary: Summary of analysis
        """
        issues = []
        actions = []  # Concrete actions for the fixer
        action_plan = []
        decisions = []
        per_image_actions = []  # Per-image decisions with reasons
        excluded_images = []  # Images to exclude from cleaned dataset
        
        # Get class labels for each image (needed for class balance checking)
        # Extract class name from image paths (usually second-to-last directory)
        class_labels = []
        for img_path in image_paths:
            path_obj = Path(img_path)
            # Find class name (parent directory name)
            if len(path_obj.parts) >= 2:
                class_labels.append(path_obj.parts[-2])
            else:
                class_labels.append('unknown')
        
        # Analyze blur with realistic handling (CRITICAL: blur is irreversible)
        if len(blur_scores) > 0:
            # Categorize blur severity
            # Anything below BLUR_THRESHOLD (100.0) is considered blurry and will be removed
            severely_blurred = [i for i, score in enumerate(blur_scores) if score < self.BLUR_THRESHOLD]
            mildly_blurred = []  # No mild category - either acceptable or removed
            acceptable_blur = [i for i, score in enumerate(blur_scores) if score >= self.BLUR_THRESHOLD]
            
            severe_count = len(severely_blurred)
            mild_count = len(mildly_blurred)
            severe_percentage = (severe_count / len(blur_scores)) * 100
            mild_percentage = (mild_count / len(blur_scores)) * 100
            
            if severe_count > 0:
                # Check class balance before excluding
                class_removal_counts = {}
                for idx in severely_blurred:
                    class_name = class_labels[idx]
                    class_removal_counts[class_name] = class_removal_counts.get(class_name, 0) + 1
                
                # Check if exclusion would cause severe imbalance
                final_excluded = []
                for idx in severely_blurred:
                    class_name = class_labels[idx]
                    class_total = class_distribution.get(class_name, 0)
                    removal_ratio = class_removal_counts.get(class_name, 0) / class_total if class_total > 0 else 0
                    
                    if removal_ratio > self.MAX_CLASS_REMOVAL_RATIO:
                        # Too many from this class - keep it but flag as problematic
                        per_image_actions.append({
                            'image_index': idx,
                            'image_path': image_paths[idx],
                            'action': 'keep_flagged',
                            'reason': f'Severely blurred but keeping to maintain class balance (would remove {removal_ratio*100:.1f}% of class)',
                            'blur_score': blur_scores[idx]
                        })
                    else:
                        # Safe to exclude
                        final_excluded.append(idx)
                        excluded_images.append({
                            'image_index': idx,
                            'image_path': image_paths[idx],
                            'reason': f'Blurry image (score: {blur_scores[idx]:.2f} < {self.BLUR_THRESHOLD}) - irreversible defect',
                            'blur_score': blur_scores[idx],
                            'class': class_name
                        })
                        per_image_actions.append({
                            'image_index': idx,
                            'image_path': image_paths[idx],
                            'action': 'exclude',
                            'reason': f'Severely blurred (score: {blur_scores[idx]:.2f}) - irreversible defect',
                            'blur_score': blur_scores[idx]
                        })
                
                if len(final_excluded) > 0:
                    issues.append({
                        'type': 'blur_severe',
                        'severity': 'critical',
                        'description': f'{len(final_excluded)} images are severely blurred and will be excluded',
                        'affected_count': len(final_excluded)
                    })
                    decisions.append(f"{len(final_excluded)} severely blurred images → EXCLUDING from cleaned dataset (blur is irreversible)")
                    action_plan.append(f"EXCLUDE {len(final_excluded)} blurry images (blur score < {self.BLUR_THRESHOLD})")
                    
                    actions.append({
                        'type': 'exclude',
                        'indices': final_excluded,
                        'reason': f'Severe blur detected - irreversible defect'
                    })
            
            if mild_count > 0:
                issues.append({
                    'type': 'blur_mild',
                    'severity': 'medium',
                    'description': f'{mild_count} images are mildly blurred ({mild_percentage:.1f}%)',
                    'affected_count': mild_count
                })
                decisions.append(f"{mild_count} mildly blurred images → keeping but flagged")
                action_plan.append(f"KEEP {mild_count} mildly blurred images (flagged for review)")
                
                for idx in mildly_blurred:
                    per_image_actions.append({
                        'image_index': idx,
                        'image_path': image_paths[idx],
                        'action': 'keep_flagged',
                        'reason': f'Mildly blurred (score: {blur_scores[idx]:.2f}) - acceptable but flagged',
                        'blur_score': blur_scores[idx]
                    })
        
        # Analyze noise with actual scores (noise CAN be fixed) - detect ALL noisy images
        noisy_indices = []
        if len(noise_scores) > 0:
            excluded_set = set([ex['image_index'] for ex in excluded_images])
            
            # Find indices of noisy images (only those not excluded)
            noisy_indices = [i for i, score in enumerate(noise_scores) 
                           if score > self.NOISE_THRESHOLD and i not in excluded_set]
            noisy_count = len(noisy_indices)
            noise_percentage = (noisy_count / len(noise_scores)) * 100 if len(noise_scores) > 0 else 0
            
            if noisy_count > 0:
                    issues.append({
                        'type': 'noise',
                        'severity': 'high' if noise_percentage > 30 else 'medium',
                        'description': f'{len(noisy_indices)} images are noisy ({noise_percentage:.1f}%)',
                        'affected_count': len(noisy_indices)
                    })
                    decisions.append(f"{len(noisy_indices)} images are noisy → applying conservative denoising")
                    
                    # Decide on action (use bilateral for noise to preserve edges - conservative)
                    actions.append({
                        'type': 'denoise',
                        'method': 'bilateral',  # Conservative denoising to preserve details
                        'indices': noisy_indices,
                        'reason': f'Noise detected - applying conservative denoising to preserve image details'
                    })
                    action_plan.append(f"Apply conservative bilateral denoising to {len(noisy_indices)} noisy images")
                    
                    for idx in noisy_indices:
                        per_image_actions.append({
                            'image_index': idx,
                            'image_path': image_paths[idx],
                            'action': 'denoise',
                            'reason': f'High noise (score: {noise_scores[idx]:.2f}) - applying conservative denoising',
                            'noise_score': noise_scores[idx]
                        })
        
        # Analyze brightness with actual scores (brightness CAN be fixed)
        dark_indices = []
        bright_indices = []
        if len(brightness_scores) > 0:
            # Debug: Show brightness range
            min_brightness = min(brightness_scores)
            max_brightness = max(brightness_scores)
            mean_brightness = np.mean(brightness_scores)
            
            dark_count = sum(1 for score in brightness_scores if score < self.DARK_THRESHOLD)
            bright_count = sum(1 for score in brightness_scores if score > self.BRIGHT_THRESHOLD)
            dark_percentage = (dark_count / len(brightness_scores)) * 100
            bright_percentage = (bright_count / len(brightness_scores)) * 100
            
            excluded_set = set([ex['image_index'] for ex in excluded_images])
            
            # Check dark images
            if dark_count > 0:
                dark_indices = [i for i, score in enumerate(brightness_scores) 
                              if score < self.DARK_THRESHOLD and i not in excluded_set]
                
                if len(dark_indices) > 0:
                    issues.append({
                        'type': 'brightness_dark',
                        'severity': 'medium',
                        'description': f'{len(dark_indices)} images are too dark ({dark_percentage:.1f}%)',
                        'affected_count': len(dark_indices)
                    })
                    decisions.append(f"{len(dark_indices)} images too dark → applying brightness normalization")
                    
                    actions.append({
                        'type': 'brightness_normalize',
                        'target_brightness': self.TARGET_MEAN,
                        'indices': dark_indices,
                        'reason': f'Low brightness detected - normalizing toward target ({self.TARGET_MEAN})'
                    })
                    action_plan.append(f"Normalize brightness to {self.TARGET_MEAN} for {len(dark_indices)} dark images")
                    
                    for idx in dark_indices:
                        per_image_actions.append({
                            'image_index': idx,
                            'image_path': image_paths[idx],
                            'action': 'brightness_normalize',
                            'reason': f'Low brightness (score: {brightness_scores[idx]:.2f}) - normalizing to {self.TARGET_MEAN}',
                            'brightness_score': brightness_scores[idx]
                        })
            
            # Check bright images
            if bright_count > 0:
                bright_indices = [i for i, score in enumerate(brightness_scores) 
                                if score > self.BRIGHT_THRESHOLD and i not in excluded_set]
                
                if len(bright_indices) > 0:
                    issues.append({
                        'type': 'brightness_bright',
                        'severity': 'medium',
                        'description': f'{len(bright_indices)} images are too bright ({bright_percentage:.1f}%)',
                        'affected_count': len(bright_indices)
                    })
                    decisions.append(f"{len(bright_indices)} images too bright → applying brightness normalization")
                    
                    actions.append({
                        'type': 'brightness_normalize',
                        'target_brightness': self.TARGET_MEAN,
                        'indices': bright_indices,
                        'reason': f'High brightness detected - normalizing toward target ({self.TARGET_MEAN})'
                    })
                    action_plan.append(f"Normalize brightness to {self.TARGET_MEAN} for {len(bright_indices)} bright images")
                    
                    for idx in bright_indices:
                        per_image_actions.append({
                            'image_index': idx,
                            'image_path': image_paths[idx],
                            'action': 'brightness_normalize',
                            'reason': f'High brightness (score: {brightness_scores[idx]:.2f}) - normalizing to {self.TARGET_MEAN}',
                            'brightness_score': brightness_scores[idx]
                        })
        
        # Analyze contrast with actual scores
        if contrast_scores and len(contrast_scores) > 0:
            excluded_set = set([ex['image_index'] for ex in excluded_images])
            
            low_contrast_count = sum(1 for score in contrast_scores if score < self.LOW_CONTRAST_THRESHOLD)
            high_contrast_count = sum(1 for score in contrast_scores if score > self.HIGH_CONTRAST_THRESHOLD)
            low_contrast_percentage = (low_contrast_count / len(contrast_scores)) * 100
            high_contrast_percentage = (high_contrast_count / len(contrast_scores)) * 100
            
            # Check low contrast
            if low_contrast_count > 0:
                low_contrast_indices = [i for i, score in enumerate(contrast_scores)
                                      if score < self.LOW_CONTRAST_THRESHOLD and i not in excluded_set]
                
                if len(low_contrast_indices) > 0:
                    issues.append({
                        'type': 'contrast_low',
                        'severity': 'medium',
                        'description': f'{len(low_contrast_indices)} images have low contrast ({low_contrast_percentage:.1f}%)',
                        'affected_count': len(low_contrast_indices)
                    })
                    decisions.append(f"{len(low_contrast_indices)} images have low contrast → applying CLAHE enhancement")
                    
                    actions.append({
                        'type': 'contrast_enhance',
                        'method': 'CLAHE',
                        'indices': low_contrast_indices,
                        'reason': f'Low contrast detected - applying CLAHE enhancement'
                    })
                    action_plan.append(f"Apply CLAHE contrast enhancement to {len(low_contrast_indices)} images")
            
            # Check high contrast
            if high_contrast_count > 0:
                high_contrast_indices = [i for i, score in enumerate(contrast_scores)
                                       if score > self.HIGH_CONTRAST_THRESHOLD and i not in excluded_set]
                
                if len(high_contrast_indices) > 0:
                    issues.append({
                        'type': 'contrast_high',
                        'severity': 'medium',
                        'description': f'{len(high_contrast_indices)} images have high contrast ({high_contrast_percentage:.1f}%)',
                        'affected_count': len(high_contrast_indices)
                    })
                    decisions.append(f"{len(high_contrast_indices)} images have high contrast → applying contrast reduction")
                    
                    actions.append({
                        'type': 'contrast_reduce',
                        'indices': high_contrast_indices,
                        'reason': f'High contrast detected - applying histogram normalization'
                    })
                    action_plan.append(f"Reduce contrast for {len(high_contrast_indices)} images")
        
        # Analyze sharpness (over-sharpened images)
        if len(blur_scores) > 0:
            excluded_set = set([ex['image_index'] for ex in excluded_images])
            over_sharpened = [i for i, score in enumerate(blur_scores)
                            if score > self.SHARPNESS_THRESHOLD and i not in excluded_set]
            over_sharpened_count = len(over_sharpened)
            over_sharpened_percentage = (over_sharpened_count / len(blur_scores)) * 100 if len(blur_scores) > 0 else 0
            
            if over_sharpened_count > 0:
                issues.append({
                    'type': 'sharpness',
                    'severity': 'medium',
                    'description': f'{over_sharpened_count} images are over-sharpened ({over_sharpened_percentage:.1f}%)',
                    'affected_count': over_sharpened_count
                })
                decisions.append(f"{over_sharpened_count} images are over-sharpened → applying Gaussian smoothing")
                
                actions.append({
                    'type': 'normalize_sharpness',
                    'indices': over_sharpened,
                    'reason': f'Over-sharpening detected - applying Gaussian smoothing to normalize'
                })
                action_plan.append(f"Normalize sharpness for {over_sharpened_count} over-sharpened images")
        
        # Analyze saturation issues
        if saturation_scores and len(saturation_scores) > 0:
            excluded_set = set([ex['image_index'] for ex in excluded_images])
            low_sat_indices = [i for i, score in enumerate(saturation_scores) 
                             if score < self.LOW_SATURATION_THRESHOLD and i not in excluded_set]
            high_sat_indices = [i for i, score in enumerate(saturation_scores) 
                              if score > self.HIGH_SATURATION_THRESHOLD and i not in excluded_set]
            
            if len(low_sat_indices) > 0:
                issues.append({
                    'type': 'low_saturation',
                    'severity': 'low',
                    'description': f'{len(low_sat_indices)} images are under-saturated',
                    'affected_count': len(low_sat_indices)
                })
                decisions.append(f"{len(low_sat_indices)} images are under-saturated → increasing saturation")
                
                actions.append({
                    'type': 'adjust_saturation',
                    'indices': low_sat_indices,
                    'saturation_factor': 1.3,  # Increase by 30%
                    'reason': f'Low saturation detected - increasing saturation'
                })
                action_plan.append(f"Increase saturation for {len(low_sat_indices)} under-saturated images")
            
            if len(high_sat_indices) > 0:
                issues.append({
                    'type': 'high_saturation',
                    'severity': 'medium',
                    'description': f'{len(high_sat_indices)} images are over-saturated',
                    'affected_count': len(high_sat_indices)
                })
                decisions.append(f"{len(high_sat_indices)} images are over-saturated → reducing saturation")
                
                actions.append({
                    'type': 'adjust_saturation',
                    'indices': high_sat_indices,
                    'saturation_factor': 0.6,  # Reduce by 40% (more aggressive)
                    'reason': f'High saturation detected - reducing saturation'
                })
                action_plan.append(f"Reduce saturation for {len(high_sat_indices)} over-saturated images")
        
        # Analyze class distribution after projected exclusions and optionally auto-balance.
        if len(class_distribution) > 1:
            projected_distribution = {
                class_name: max(0, count)
                for class_name, count in class_distribution.items()
            }
            for excluded in excluded_images:
                class_name = excluded.get('class')
                if class_name in projected_distribution:
                    projected_distribution[class_name] = max(
                        0, projected_distribution[class_name] - 1
                    )

            non_empty_distribution = {
                class_name: count
                for class_name, count in projected_distribution.items()
                if count > 0
            }

            if len(non_empty_distribution) > 1:
                counts = list(non_empty_distribution.values())
                min_count = min(counts)
                max_count = max(counts)

                if max_count - min_count > self.BALANCE_TOLERANCE:
                    if balance_strategy == 'undersample':
                        target_count = min_count
                    elif balance_strategy == 'oversample':
                        target_count = max_count
                    else:
                        target_count = max(1, int(round(float(np.median(counts)))))

                    issues.append({
                        'type': 'class_imbalance',
                        'severity': 'high',
                        'description': (
                            f'Classes are imbalanced after cleaning '
                            f'(min={min_count}, max={max_count}, target={target_count})'
                        ),
                        'affected_count': max_count - min_count
                    })

                    if auto_balance:
                        decisions.append(
                            f'Class imbalance detected -> auto-balancing dataset with {balance_strategy} strategy'
                        )
                        action_plan.append(
                            f"Balance dataset automatically using {balance_strategy} strategy to target {target_count} images per class"
                        )
                        actions.append({
                            'type': 'balance_dataset',
                            'strategy': balance_strategy,
                            'target_count': target_count,
                            'projected_distribution': non_empty_distribution,
                            'reason': (
                                f'Class imbalance detected after quality fixes - '
                                f'balancing cleaned dataset to {target_count} images per class'
                            )
                        })
                    else:
                        decisions.append(
                            f'Class imbalance detected -> recommend running auto-balance ({balance_strategy})'
                        )
                        action_plan.append(
                            f"RECOMMENDATION: Re-run with --fix --balance --balance-strategy {balance_strategy}"
                        )
        
        # Track images that are kept unchanged (good quality)
        excluded_set = set([ex['image_index'] for ex in excluded_images])
        fixed_set = set()
        for action in actions:
            if action['type'] != 'exclude':
                fixed_set.update(action.get('indices', []))
        
        kept_unchanged = []
        for i in range(len(image_paths)):
            if i not in excluded_set and i not in fixed_set:
                per_image_actions.append({
                    'image_index': i,
                    'image_path': image_paths[i],
                    'action': 'keep',
                    'reason': 'Image quality is acceptable - keeping unchanged',
                    'blur_score': blur_scores[i] if i < len(blur_scores) else None,
                    'noise_score': noise_scores[i] if i < len(noise_scores) else None,
                    'brightness_score': brightness_scores[i] if i < len(brightness_scores) else None
                })
        
        return {
            'issues': issues,
            'actions': actions,  # Concrete actions for fixer
            'action_plan': action_plan,  # Human-readable plan
            'decisions': decisions,
            'per_image_actions': per_image_actions,  # Detailed per-image decisions
            'excluded_images': excluded_images,  # Images excluded from cleaned dataset
            'summary': self._generate_summary(issues, decisions, len(excluded_images))
        }
    
    def analyze_with_scores(self, blur_scores: List[float], 
                           noise_scores: List[float],
                           brightness_scores: List[float],
                           class_distribution: Dict[str, int]) -> Dict:
        """
        More accurate analysis using individual image scores.
        This is the legacy method that returns recommendations (not actions).
        Use decide_actions() for actionable decisions.
        
        Args:
            blur_scores: List of blur scores for each image
            noise_scores: List of noise scores for each image
            brightness_scores: List of brightness scores for each image
            class_distribution: Dictionary mapping class names to counts
            
        Returns:
            Dictionary containing detected issues and recommendations
        """
        issues = []
        recommendations = []
        decisions = []
        
        # Analyze blur with actual scores
        if len(blur_scores) > 0:
            blurry_count = sum(1 for score in blur_scores if score < self.BLUR_THRESHOLD)
            blur_percentage = (blurry_count / len(blur_scores)) * 100
            
            if blur_percentage > 10:  # More than 10% blurry
                issues.append({
                    'type': 'blur',
                    'severity': 'high' if blur_percentage > 30 else 'medium',
                    'description': f'{blur_percentage:.1f}% of images are blurry ({blurry_count}/{len(blur_scores)})'
                })
                decisions.append(f"{blur_percentage:.0f}% images are blurry → recommend denoising")
                recommendations.append("Apply denoising filter to blurry images")
        
        # Analyze noise with actual scores
        if len(noise_scores) > 0:
            noisy_count = sum(1 for score in noise_scores if score > self.NOISE_THRESHOLD)
            noise_percentage = (noisy_count / len(noise_scores)) * 100
            
            if noise_percentage > 10:  # More than 10% noisy
                issues.append({
                    'type': 'noise',
                    'severity': 'high' if noise_percentage > 30 else 'medium',
                    'description': f'{noise_percentage:.1f}% of images are noisy ({noisy_count}/{len(noise_scores)})'
                })
                decisions.append(f"{noise_percentage:.0f}% images are noisy → recommend denoising")
                recommendations.append("Apply noise reduction filter")
        
        # Analyze brightness with actual scores
        if len(brightness_scores) > 0:
            dark_count = sum(1 for score in brightness_scores if score < self.DARK_THRESHOLD)
            bright_count = sum(1 for score in brightness_scores if score > self.BRIGHT_THRESHOLD)
            dark_percentage = (dark_count / len(brightness_scores)) * 100
            bright_percentage = (bright_count / len(brightness_scores)) * 100
            
            if dark_percentage > 20:  # More than 20% too dark
                issues.append({
                    'type': 'brightness',
                    'severity': 'medium',
                    'description': f'{dark_percentage:.1f}% of images are too dark'
                })
                decisions.append("Dataset too dark → recommend brightness normalization")
                recommendations.append("Apply brightness normalization to improve visibility")
            
            if bright_percentage > 20:  # More than 20% too bright
                issues.append({
                    'type': 'brightness',
                    'severity': 'medium',
                    'description': f'{bright_percentage:.1f}% of images are too bright'
                })
                decisions.append("Dataset too bright → recommend brightness normalization")
                recommendations.append("Apply brightness normalization to reduce overexposure")
        
        # Analyze class distribution
        if len(class_distribution) > 0:
            class_counts = list(class_distribution.values())
            total_images = sum(class_counts)
            avg_count = total_images / len(class_distribution)
            
            imbalanced_classes = []
            for class_name, count in class_distribution.items():
                ratio = count / avg_count if avg_count > 0 else 0
                if ratio < self.CLASS_IMBALANCE_RATIO:
                    imbalanced_classes.append((class_name, count, ratio))
            
            if imbalanced_classes:
                for class_name, count, ratio in imbalanced_classes:
                    issues.append({
                        'type': 'class_imbalance',
                        'severity': 'high',
                        'description': f'Class "{class_name}" is under-represented ({count} images, {ratio*100:.1f}% of average)'
                    })
                    decisions.append(f'Class "{class_name}" is under-represented → recommend oversampling')
                    recommendations.append(f"Apply oversampling or data augmentation for class '{class_name}'")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'decisions': decisions,
            'summary': self._generate_summary(issues, decisions)
        }
