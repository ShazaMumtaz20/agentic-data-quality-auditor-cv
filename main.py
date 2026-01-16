"""
Main Entry Point
Example usage of the Agentic Data Quality Auditor for Computer Vision.
"""

import argparse
from pathlib import Path
from data_loader import DataLoader
from quality_metrics import QualityMetrics
from agent import QualityAgent
from visualizer import Visualizer
from fixer import ImageFixer
from report_generator import ReportGenerator
from preprocessor import DatasetPreprocessor


def main():
    """
    Main function to run the data quality audit.
    """
    parser = argparse.ArgumentParser(
        description='Agentic Data Quality Auditor for Computer Vision'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the dataset directory (ImageFolder structure)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for reports and visualizations (default: output)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Apply auto-fixes to images (saves to fixed_images/)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip generating visualizations'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AGENTIC DATA QUALITY AUDITOR FOR COMPUTER VISION")
    print("=" * 80)
    print()
    
    # Step 1: Load dataset
    print("Step 1: Loading dataset...")
    try:
        data_loader = DataLoader(args.dataset)
        dataset_dict = data_loader.load_dataset()
        class_distribution = data_loader.get_class_distribution()
        total_images = data_loader.get_total_images()
        
        print(f"  ✓ Loaded {total_images} images from {len(class_distribution)} classes")
        print(f"  Classes: {', '.join(sorted(class_distribution.keys()))}")
        print()
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return
    
    # Step 2: Compute quality metrics
    print("Step 2: Computing quality metrics...")
    try:
        quality_metrics = QualityMetrics(data_loader)
        metrics_data = quality_metrics.analyze_dataset()
        metrics_stats = quality_metrics.get_statistics()
        print()
    except Exception as e:
        print(f"  ✗ Error computing metrics: {e}")
        return
    
    # Step 3: Agent analysis and decision making
    print("Step 3: Running agent analysis...")
    try:
        agent = QualityAgent()
        
        # Use decide_actions for actionable decisions (when --fix is used)
        if args.fix:
            # Get all available metrics
            contrast_scores = metrics_data.get('contrast_scores', [])
            saturation_scores = metrics_data.get('saturation_scores', [])
            corruption_flags = metrics_data.get('corruption_flags', [False] * len(metrics_data['image_paths']))
            
            agent_analysis = agent.decide_actions(
                blur_scores=metrics_data['blur_scores'],
                noise_scores=metrics_data['noise_scores'],
                brightness_scores=metrics_data['brightness_scores'],
                image_paths=metrics_data['image_paths'],
                class_distribution=class_distribution,
                contrast_scores=contrast_scores if contrast_scores else None,
                saturation_scores=saturation_scores if saturation_scores else None,
                corruption_flags=corruption_flags if corruption_flags else None
            )
            
            print("  Agent Summary:")
            print(f"  {agent_analysis['summary']}")
            if len(agent_analysis['action_plan']) > 0:
                print("\n  Action Plan:")
                for action in agent_analysis['action_plan']:
                    print(f"    • {action}")
            if len(agent_analysis['decisions']) > 0:
                print("\n  Decisions:")
                for decision in agent_analysis['decisions']:
                    print(f"    • {decision}")
        else:
            # Use analyze_with_scores for recommendations only
            agent_analysis = agent.analyze_with_scores(
                blur_scores=metrics_data['blur_scores'],
                noise_scores=metrics_data['noise_scores'],
                brightness_scores=metrics_data['brightness_scores'],
                class_distribution=class_distribution
            )
            
            print("  Agent Summary:")
            print(f"  {agent_analysis['summary']}")
            if len(agent_analysis['decisions']) > 0:
                print("\n  Decisions:")
                for decision in agent_analysis['decisions']:
                    print(f"    • {decision}")
        print()
    except Exception as e:
        print(f"  ✗ Error in agent analysis: {e}")
        return
    
    # Store before metrics for comparison
    before_metrics = {
        'stats': metrics_stats,
        'data': metrics_data,
        'class_distribution': class_distribution,
        'total_images': total_images
    }
    
    # Step 4: Generate visualizations
    if not args.no_viz:
        print("Step 4: Generating visualizations...")
        try:
            visualizer = Visualizer(output_dir=args.output)
            visualizer.generate_all_plots(
                brightness_scores=metrics_data['brightness_scores'],
                blur_scores=metrics_data['blur_scores'],
                noise_scores=metrics_data['noise_scores'],
                class_distribution=class_distribution
            )
            print()
        except Exception as e:
            print(f"  ✗ Error generating visualizations: {e}")
            print()
    
    # Step 5: Apply fixes (if requested)
    fixed_count = 0
    after_metrics = None
    if args.fix:
        print("Step 5: Applying agent-selected fixes...")
        try:
            fixer = ImageFixer(data_loader, output_dir="cleaned_dataset")
            
            # Apply fixes based on agent actions
            if 'actions' in agent_analysis and len(agent_analysis['actions']) > 0:
                fixed_count, skipped, excluded_count, fixes_log = fixer.apply_agent_actions(
                    actions=agent_analysis['actions'],
                    image_paths=metrics_data['image_paths']
                )
                
                print(f"\n  ✓ Applied fixes to {fixed_count} images")
                print(f"  ✓ Copied {skipped} unchanged images")
                if excluded_count > 0:
                    print(f"  ⚠ Excluded {excluded_count} images (irreversible defects)")
                print(f"  ✓ Cleaned dataset saved to: cleaned_dataset/")
                
                # Show excluded images summary if any
                if 'excluded_images' in agent_analysis and len(agent_analysis['excluded_images']) > 0:
                    print(f"\n  Excluded Images Summary:")
                    excluded_by_class = {}
                    for ex in agent_analysis['excluded_images']:
                        class_name = ex.get('class', 'unknown')
                        excluded_by_class[class_name] = excluded_by_class.get(class_name, 0) + 1
                    for class_name, count in excluded_by_class.items():
                        print(f"    - {class_name}: {count} images excluded")
                
                # Step 5b: Re-analyze cleaned dataset
                print("\nStep 5b: Re-analyzing cleaned dataset...")
                try:
                    cleaned_loader = DataLoader("cleaned_dataset")
                    cleaned_loader.load_dataset()
                    cleaned_class_dist = cleaned_loader.get_class_distribution()
                    
                    cleaned_quality_metrics = QualityMetrics(cleaned_loader)
                    cleaned_metrics_data = cleaned_quality_metrics.analyze_dataset()
                    cleaned_metrics_stats = cleaned_quality_metrics.get_statistics()
                    
                    after_metrics = {
                        'stats': cleaned_metrics_stats,
                        'data': cleaned_metrics_data,
                        'class_distribution': cleaned_class_dist,
                        'total_images': cleaned_loader.get_total_images()
                    }
                    
                    # Compare before and after
                    print("\n  Before vs After Comparison:")
                    print("  " + "-" * 76)
                    
                    # Blur comparison
                    if 'blur' in before_metrics['stats'] and 'blur' in after_metrics['stats']:
                        before_blur = before_metrics['stats']['blur']['mean']
                        after_blur = after_metrics['stats']['blur']['mean']
                        improvement = ((after_blur - before_blur) / before_blur) * 100 if before_blur > 0 else 0
                        print(f"  Blur Score:     {before_blur:.2f} → {after_blur:.2f} ({improvement:+.1f}%)")
                    
                    # Noise comparison
                    if 'noise' in before_metrics['stats'] and 'noise' in after_metrics['stats']:
                        before_noise = before_metrics['stats']['noise']['mean']
                        after_noise = after_metrics['stats']['noise']['mean']
                        improvement = ((before_noise - after_noise) / before_noise) * 100 if before_noise > 0 else 0
                        print(f"  Noise Score:     {before_noise:.2f} → {after_noise:.2f} ({improvement:+.1f}% reduction)")
                    
                    # Brightness comparison
                    if 'brightness' in before_metrics['stats'] and 'brightness' in after_metrics['stats']:
                        before_bright = before_metrics['stats']['brightness']['mean']
                        after_bright = after_metrics['stats']['brightness']['mean']
                        improvement = abs(after_bright - 128.0) < abs(before_bright - 128.0)
                        print(f"  Brightness:      {before_bright:.2f} → {after_bright:.2f} (target: 128.0)")
                        if improvement:
                            print(f"    ✓ Closer to target brightness")
                    
                    print()
                    
                except Exception as e:
                    print(f"  ✗ Error re-analyzing cleaned dataset: {e}")
                    print("  (Continuing with before metrics only)")
            else:
                print("  No actions to apply - dataset quality is acceptable")
            print()
        except Exception as e:
            print(f"  ✗ Error applying fixes: {e}")
            print()
    
    # Step 6: Generate reports
    print("Step 6: Generating reports...")
    try:
        report_generator = ReportGenerator(output_dir=args.output)
        
        # Use after metrics if available, otherwise use before
        final_metrics_stats = after_metrics['stats'] if after_metrics else metrics_stats
        final_class_dist = after_metrics['class_distribution'] if after_metrics else class_distribution
        final_total = after_metrics['total_images'] if after_metrics else total_images
        
        # Generate text report with before/after comparison
        report_generator.generate_text_report(
            dataset_path=args.dataset,
            metrics_stats=final_metrics_stats,
            class_distribution=final_class_dist,
            agent_analysis=agent_analysis,
            total_images=final_total,
            fixed_count=fixed_count,
            before_metrics=before_metrics if args.fix and after_metrics else None,
            after_metrics=after_metrics if args.fix else None
        )
        
        # Generate JSON report
        report_generator.generate_json_report(
            dataset_path=args.dataset,
            metrics_stats=final_metrics_stats,
            class_distribution=final_class_dist,
            agent_analysis=agent_analysis,
            total_images=final_total,
            fixed_count=fixed_count,
            before_metrics=before_metrics if args.fix and after_metrics else None,
            after_metrics=after_metrics if args.fix else None
        )
        print()
    except Exception as e:
        print(f"  ✗ Error generating reports: {e}")
        print()
    
    # Final summary
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Reports saved to: {args.output}/")
    if args.fix:
        print(f"Cleaned dataset saved to: cleaned_dataset/")
        if after_metrics:
            print(f"Before/after comparison included in reports")
    print()


if __name__ == "__main__":
    # Example usage without command-line arguments (for testing)
    # Uncomment and modify the path below to test directly
    
    # dataset_path = "path/to/your/dataset"
    # data_loader = DataLoader(dataset_path)
    # quality_metrics = QualityMetrics(data_loader)
    # metrics_data = quality_metrics.analyze_dataset()
    # metrics_stats = quality_metrics.get_statistics()
    # class_distribution = quality_metrics.get_class_distribution()
    # 
    # agent = QualityAgent()
    # agent_analysis = agent.analyze_with_scores(
    #     metrics_data['blur_scores'],
    #     metrics_data['noise_scores'],
    #     metrics_data['brightness_scores'],
    #     class_distribution
    # )
    # 
    # visualizer = Visualizer()
    # visualizer.generate_all_plots(
    #     metrics_data['brightness_scores'],
    #     metrics_data['blur_scores'],
    #     metrics_data['noise_scores'],
    #     class_distribution
    # )
    # 
    # report_generator = ReportGenerator()
    # report_generator.generate_text_report(
    #     dataset_path,
    #     metrics_stats,
    #     class_distribution,
    #     agent_analysis,
    #     len(metrics_data['image_paths'])
    # )
    
    main()
