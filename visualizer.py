"""
Visualization Module
Generates plots and charts for dataset quality analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from pathlib import Path


class Visualizer:
    """
    Creates visualizations for dataset quality analysis.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_brightness_histogram(self, brightness_scores: List[float], save_path: str = None):
        """
        Plot histogram of image brightness distribution.
        
        Args:
            brightness_scores: List of brightness values
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.hist(brightness_scores, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Brightness (Mean Pixel Intensity)', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Distribution of Image Brightness', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for thresholds
        plt.axvline(x=50, color='red', linestyle='--', label='Dark Threshold (50)')
        plt.axvline(x=200, color='orange', linestyle='--', label='Bright Threshold (200)')
        plt.legend()
        
        if save_path is None:
            save_path = self.output_dir / "brightness_histogram.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Brightness histogram saved to {save_path}")
    
    def plot_class_distribution(self, class_distribution: Dict[str, int], save_path: str = None):
        """
        Plot bar chart of class distribution.
        
        Args:
            class_distribution: Dictionary mapping class names to counts
            save_path: Optional path to save the plot
        """
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(classes, counts, edgecolor='black', alpha=0.7)
        
        # Color bars based on whether they're imbalanced
        avg_count = np.mean(counts)
        for i, (class_name, count) in enumerate(class_distribution.items()):
            if count < avg_count * 0.3:
                bars[i].set_color('red')
            elif count < avg_count * 0.7:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('green')
        
        plt.xlabel('Class Name', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add average line
        plt.axhline(y=avg_count, color='blue', linestyle='--', label=f'Average ({avg_count:.0f})')
        plt.legend()
        
        if save_path is None:
            save_path = self.output_dir / "class_distribution.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class distribution chart saved to {save_path}")
    
    def plot_blur_distribution(self, blur_scores: List[float], save_path: str = None):
        """
        Plot distribution of blur scores.
        
        Args:
            blur_scores: List of blur scores (Laplacian variance)
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.hist(blur_scores, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Blur Score (Laplacian Variance)', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Distribution of Blur Scores', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line for threshold
        plt.axvline(x=100, color='red', linestyle='--', label='Blur Threshold (100)')
        plt.legend()
        
        if save_path is None:
            save_path = self.output_dir / "blur_distribution.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Blur distribution plot saved to {save_path}")
    
    def plot_noise_distribution(self, noise_scores: List[float], save_path: str = None):
        """
        Plot distribution of noise scores.
        
        Args:
            noise_scores: List of noise scores
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.hist(noise_scores, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Noise Score', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Distribution of Noise Scores', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line for threshold
        plt.axvline(x=20, color='red', linestyle='--', label='Noise Threshold (20)')
        plt.legend()
        
        if save_path is None:
            save_path = self.output_dir / "noise_distribution.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Noise distribution plot saved to {save_path}")
    
    def generate_all_plots(self, brightness_scores: List[float],
                          blur_scores: List[float],
                          noise_scores: List[float],
                          class_distribution: Dict[str, int]):
        """
        Generate all visualization plots.
        
        Args:
            brightness_scores: List of brightness values
            blur_scores: List of blur scores
            noise_scores: List of noise scores
            class_distribution: Dictionary mapping class names to counts
        """
        print("\nGenerating visualizations...")
        self.plot_brightness_histogram(brightness_scores)
        self.plot_class_distribution(class_distribution)
        self.plot_blur_distribution(blur_scores)
        self.plot_noise_distribution(noise_scores)
        print("All visualizations generated successfully!")
