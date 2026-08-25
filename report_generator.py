"""
Report Generation Module
Generates text and PDF reports for dataset quality analysis.
"""

from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json


class ReportGenerator:
    """
    Generates quality audit reports in text and PDF formats.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_text_report(self, dataset_path: str,
                           metrics_stats: Dict,
                           class_distribution: Dict[str, int],
                           agent_analysis: Dict,
                           total_images: int,
                           fixed_count: int = 0,
                           before_metrics: Dict = None,
                           after_metrics: Dict = None,
                           save_path: str = None) -> str:
        """
        Generate a text report.
        
        Args:
            dataset_path: Path to the dataset
            metrics_stats: Statistics from QualityMetrics
            class_distribution: Class distribution dictionary
            agent_analysis: Analysis results from QualityAgent
            total_images: Total number of images
            fixed_count: Number of images that were fixed
            save_path: Optional path to save the report
            
        Returns:
            Report content as string
        """
        report = []
        report.append("=" * 80)
        report.append("AGENTIC DATA QUALITY AUDITOR FOR COMPUTER VISION")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset Summary
        report.append("DATASET SUMMARY")
        report.append("-" * 80)
        report.append(f"Dataset Path: {dataset_path}")
        report.append(f"Total Images: {total_images}")
        report.append(f"Number of Classes: {len(class_distribution)}")
        report.append("")
        
        # Class Distribution
        report.append("CLASS DISTRIBUTION")
        report.append("-" * 80)
        for class_name, count in sorted(class_distribution.items()):
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            report.append(f"  {class_name}: {count} images ({percentage:.1f}%)")
        report.append("")
        
        # Quality Metrics (with before/after comparison if available)
        if before_metrics and after_metrics:
            report.append("QUALITY METRICS - BEFORE vs AFTER")
            report.append("-" * 80)
            
            if 'blur' in before_metrics['stats'] and 'blur' in after_metrics['stats']:
                before_blur = before_metrics['stats']['blur']
                after_blur = after_metrics['stats']['blur']
                improvement = ((after_blur['mean'] - before_blur['mean']) / before_blur['mean']) * 100 if before_blur['mean'] > 0 else 0
                report.append("Blur Scores (Laplacian Variance):")
                report.append(f"  Before - Mean: {before_blur['mean']:.2f}")
                report.append(f"  After  - Mean: {after_blur['mean']:.2f} ({improvement:+.1f}%)")
                report.append("")
            
            if 'noise' in before_metrics['stats'] and 'noise' in after_metrics['stats']:
                before_noise = before_metrics['stats']['noise']
                after_noise = after_metrics['stats']['noise']
                improvement = ((before_noise['mean'] - after_noise['mean']) / before_noise['mean']) * 100 if before_noise['mean'] > 0 else 0
                report.append("Noise Scores:")
                report.append(f"  Before - Mean: {before_noise['mean']:.2f}")
                report.append(f"  After  - Mean: {after_noise['mean']:.2f} ({improvement:+.1f}% reduction)")
                report.append("")
            
            if 'brightness' in before_metrics['stats'] and 'brightness' in after_metrics['stats']:
                before_bright = before_metrics['stats']['brightness']
                after_bright = after_metrics['stats']['brightness']
                report.append("Brightness Scores (Mean Pixel Intensity):")
                report.append(f"  Before - Mean: {before_bright['mean']:.2f}")
                report.append(f"  After  - Mean: {after_bright['mean']:.2f} (target: 128.0)")
                before_dist = abs(before_bright['mean'] - 128.0)
                after_dist = abs(after_bright['mean'] - 128.0)
                if after_dist < before_dist:
                    report.append(f"  ✓ Improved: {before_dist:.2f} → {after_dist:.2f} deviation from target")
                report.append("")
            
            if 'contrast' in before_metrics['stats'] and 'contrast' in after_metrics['stats']:
                before_contrast = before_metrics['stats']['contrast']
                after_contrast = after_metrics['stats']['contrast']
                improvement = ((after_contrast['mean'] - before_contrast['mean']) / before_contrast['mean']) * 100 if before_contrast['mean'] > 0 else 0
                report.append("Contrast Scores (Standard Deviation):")
                report.append(f"  Before - Mean: {before_contrast['mean']:.2f}")
                report.append(f"  After  - Mean: {after_contrast['mean']:.2f} ({improvement:+.1f}%)")
                report.append("")
            
            if 'sharpness' in before_metrics['stats'] and 'sharpness' in after_metrics['stats']:
                before_sharp = before_metrics['stats']['sharpness']
                after_sharp = after_metrics['stats']['sharpness']
                report.append("Sharpness Scores (Laplacian Variance):")
                report.append(f"  Before - Mean: {before_sharp['mean']:.2f}")
                report.append(f"  After  - Mean: {after_sharp['mean']:.2f}")
                report.append("")
        else:
            report.append("QUALITY METRICS")
            report.append("-" * 80)
            
            if 'blur' in metrics_stats:
                blur = metrics_stats['blur']
                report.append("Blur Scores (Laplacian Variance):")
                report.append(f"  Mean: {blur['mean']:.2f}")
                report.append(f"  Std:  {blur['std']:.2f}")
                report.append(f"  Min:  {blur['min']:.2f}")
                report.append(f"  Max:  {blur['max']:.2f}")
                report.append(f"  Median: {blur['median']:.2f}")
                report.append("")
            
            if 'noise' in metrics_stats:
                noise = metrics_stats['noise']
                report.append("Noise Scores:")
                report.append(f"  Mean: {noise['mean']:.2f}")
                report.append(f"  Std:  {noise['std']:.2f}")
                report.append(f"  Min:  {noise['min']:.2f}")
                report.append(f"  Max:  {noise['max']:.2f}")
                report.append(f"  Median: {noise['median']:.2f}")
                report.append("")
            
            if 'brightness' in metrics_stats:
                brightness = metrics_stats['brightness']
                report.append("Brightness Scores (Mean Pixel Intensity):")
                report.append(f"  Mean: {brightness['mean']:.2f}")
                report.append(f"  Std:  {brightness['std']:.2f}")
                report.append(f"  Min:  {brightness['min']:.2f}")
                report.append(f"  Max:  {brightness['max']:.2f}")
                report.append(f"  Median: {brightness['median']:.2f}")
                report.append("")
        
        # Agent Analysis
        report.append("AGENT ANALYSIS")
        report.append("-" * 80)
        report.append(agent_analysis['summary'])
        report.append("")
        
        if 'action_plan' in agent_analysis and len(agent_analysis['action_plan']) > 0:
            report.append("Action Plan:")
            for i, action in enumerate(agent_analysis['action_plan'], 1):
                report.append(f"  {i}. {action}")
            report.append("")
        
        if len(agent_analysis['decisions']) > 0:
            report.append("Agent Decisions:")
            for i, decision in enumerate(agent_analysis['decisions'], 1):
                report.append(f"  {i}. {decision}")
            report.append("")
        
        if 'recommendations' in agent_analysis and len(agent_analysis['recommendations']) > 0:
            report.append("Recommendations:")
            for i, rec in enumerate(agent_analysis['recommendations'], 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        # Detected Issues
        if len(agent_analysis['issues']) > 0:
            report.append("DETECTED ISSUES")
            report.append("-" * 80)
            for i, issue in enumerate(agent_analysis['issues'], 1):
                report.append(f"{i}. [{issue['severity'].upper()}] {issue['type'].upper()}")
                report.append(f"   {issue['description']}")
                report.append("")
        
        # Sharpness Analysis
        if 'sharpness' in metrics_stats:
            sharpness = metrics_stats['sharpness']
            report.append("Sharpness Scores (Laplacian Variance):")
            report.append(f"  Mean: {sharpness['mean']:.2f}")
            report.append(f"  Std:  {sharpness['std']:.2f}")
            report.append(f"  Min:  {sharpness['min']:.2f}")
            report.append(f"  Max:  {sharpness['max']:.2f}")
            report.append(f"  Median: {sharpness['median']:.2f}")
            report.append("")
        
        # Model Evaluation Results
        if 'evaluation' in agent_analysis:
            report.append("MODEL EVALUATION RESULTS")
            report.append("-" * 80)
            eval_data = agent_analysis['evaluation']
            if 'model_type' in eval_data:
                report.append(f"Model: {eval_data['model_type']}")
            report.append(f"Original Dataset Accuracy: {eval_data['original_accuracy']:.2f}%")
            report.append(f"Cleaned Dataset Accuracy: {eval_data['cleaned_accuracy']:.2f}%")
            if 'original_macro_f1' in eval_data and 'cleaned_macro_f1' in eval_data:
                report.append(f"Original Macro F1: {eval_data['original_macro_f1']:.4f}")
                report.append(f"Cleaned Macro F1: {eval_data['cleaned_macro_f1']:.4f}")
            report.append(f"Improvement: {eval_data['improvement']:+.2f}% ({eval_data['improvement_percent']:+.1f}% relative)")
            if 'original_confusion_matrix' in eval_data:
                report.append(f"Original Confusion Matrix: {eval_data['original_confusion_matrix']}")
            if 'cleaned_confusion_matrix' in eval_data:
                report.append(f"Cleaned Confusion Matrix: {eval_data['cleaned_confusion_matrix']}")
            if eval_data['justified']:
                report.append("✓ Preprocessing justified - cleaned dataset performs better")
            else:
                report.append("⚠ Review preprocessing - cleaned dataset performance needs improvement")
            report.append("")
        
        # Fixes Applied and Exclusions
        if fixed_count > 0 or (agent_analysis.get('excluded_images') and len(agent_analysis['excluded_images']) > 0):
            report.append("DATASET CLEANING ACTIONS")
            report.append("-" * 80)
            if fixed_count > 0:
                report.append(f"Number of images fixed: {fixed_count}")
            if agent_analysis.get('excluded_images') and len(agent_analysis['excluded_images']) > 0:
                excluded_count = len(agent_analysis['excluded_images'])
                report.append(f"Number of images EXCLUDED: {excluded_count} (irreversible defects)")
                report.append("")
                report.append("Excluded Images by Class:")
                excluded_by_class = {}
                for ex in agent_analysis['excluded_images']:
                    class_name = ex.get('class', 'unknown')
                    excluded_by_class[class_name] = excluded_by_class.get(class_name, 0) + 1
                for class_name, count in sorted(excluded_by_class.items()):
                    report.append(f"  {class_name}: {count} images")
            report.append(f"Cleaned dataset saved to: cleaned_dataset/")
            if before_metrics and after_metrics:
                report.append("")
                report.append("IMPROVEMENT SUMMARY")
                report.append("-" * 80)
                if 'blur' in before_metrics['stats'] and 'blur' in after_metrics['stats']:
                    before_blur = before_metrics['stats']['blur']['mean']
                    after_blur = after_metrics['stats']['blur']['mean']
                    improvement = ((after_blur - before_blur) / before_blur) * 100 if before_blur > 0 else 0
                    report.append(f"Blur: {before_blur:.2f} → {after_blur:.2f} ({improvement:+.1f}%)")
                if 'noise' in before_metrics['stats'] and 'noise' in after_metrics['stats']:
                    before_noise = before_metrics['stats']['noise']['mean']
                    after_noise = after_metrics['stats']['noise']['mean']
                    improvement = ((before_noise - after_noise) / before_noise) * 100 if before_noise > 0 else 0
                    report.append(f"Noise: {before_noise:.2f} → {after_noise:.2f} ({improvement:+.1f}% reduction)")
                if 'brightness' in before_metrics['stats'] and 'brightness' in after_metrics['stats']:
                    before_bright = before_metrics['stats']['brightness']['mean']
                    after_bright = after_metrics['stats']['brightness']['mean']
                    report.append(f"Brightness: {before_bright:.2f} → {after_bright:.2f} (target: 128.0)")
            report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append("End of Report")
        report.append("=" * 80)
        
        report_content = "\n".join(report)
        
        if save_path is None:
            save_path = self.output_dir / "quality_report.txt"
        else:
            save_path = Path(save_path)
        
        save_path.write_text(report_content, encoding='utf-8')
        print(f"Text report saved to {save_path}")
        
        return report_content
    
    def generate_json_report(self, dataset_path: str,
                           metrics_stats: Dict,
                           class_distribution: Dict[str, int],
                           agent_analysis: Dict,
                           total_images: int,
                           fixed_count: int = 0,
                           before_metrics: Dict = None,
                           after_metrics: Dict = None,
                           save_path: str = None) -> Dict:
        """
        Generate a JSON report for programmatic access.
        
        Args:
            dataset_path: Path to the dataset
            metrics_stats: Statistics from QualityMetrics
            class_distribution: Class distribution dictionary
            agent_analysis: Analysis results from QualityAgent
            total_images: Total number of images
            fixed_count: Number of images that were fixed
            save_path: Optional path to save the report
            
        Returns:
            Report as dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset': {
                'path': dataset_path,
                'total_images': total_images,
                'num_classes': len(class_distribution),
                'class_distribution': class_distribution
            },
            'metrics': metrics_stats,
            'agent_analysis': {
                'summary': agent_analysis['summary'],
                'issues': agent_analysis['issues'],
                'decisions': agent_analysis['decisions']
            },
            'fixes': {
                'images_fixed': fixed_count
            }
        }
        
        # Add action plan if available
        if 'action_plan' in agent_analysis:
            report['agent_analysis']['action_plan'] = agent_analysis['action_plan']

        # Add recommendations if available
        if 'recommendations' in agent_analysis:
            report['agent_analysis']['recommendations'] = agent_analysis['recommendations']

        if 'evaluation' in agent_analysis:
            report['agent_analysis']['evaluation'] = agent_analysis['evaluation']
        
        # Add before/after comparison if available
        if before_metrics and after_metrics:
            report['before_after_comparison'] = {
                'before': {
                    'metrics': before_metrics['stats'],
                    'class_distribution': before_metrics['class_distribution'],
                    'total_images': before_metrics['total_images']
                },
                'after': {
                    'metrics': after_metrics['stats'],
                    'class_distribution': after_metrics['class_distribution'],
                    'total_images': after_metrics['total_images']
                }
            }
            
            # Calculate improvements
            improvements = {}
            if 'blur' in before_metrics['stats'] and 'blur' in after_metrics['stats']:
                before_blur = before_metrics['stats']['blur']['mean']
                after_blur = after_metrics['stats']['blur']['mean']
                improvements['blur'] = {
                    'before': before_blur,
                    'after': after_blur,
                    'improvement_percent': ((after_blur - before_blur) / before_blur) * 100 if before_blur > 0 else 0
                }
            if 'noise' in before_metrics['stats'] and 'noise' in after_metrics['stats']:
                before_noise = before_metrics['stats']['noise']['mean']
                after_noise = after_metrics['stats']['noise']['mean']
                improvements['noise'] = {
                    'before': before_noise,
                    'after': after_noise,
                    'improvement_percent': ((before_noise - after_noise) / before_noise) * 100 if before_noise > 0 else 0
                }
            if 'brightness' in before_metrics['stats'] and 'brightness' in after_metrics['stats']:
                before_bright = before_metrics['stats']['brightness']['mean']
                after_bright = after_metrics['stats']['brightness']['mean']
                improvements['brightness'] = {
                    'before': before_bright,
                    'after': after_bright,
                    'target': 128.0,
                    'deviation_before': abs(before_bright - 128.0),
                    'deviation_after': abs(after_bright - 128.0)
                }
            report['before_after_comparison']['improvements'] = improvements
        
        if save_path is None:
            save_path = self.output_dir / "quality_report.json"
        else:
            save_path = Path(save_path)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"JSON report saved to {save_path}")
        
        return report
