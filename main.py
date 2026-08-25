"""
Main Entry Point
Example usage of the Agentic Data Quality Auditor for Computer Vision.
"""

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
import sys

import numpy as np

from data_loader import DataLoader
from quality_metrics import QualityMetrics
from agent import QualityAgent
from visualizer import Visualizer
from fixer import ImageFixer
from report_generator import ReportGenerator
from preprocessor import DatasetPreprocessor
# ModelEvaluator imported conditionally when --evaluate is used


TARGET_RANGES = {
    'blur': {'type': 'single', 'target': 100.0},
    'sharpness': {'type': 'single', 'target': 500.0},
    'noise': {'type': 'single', 'target': 20.0},
    'brightness': {'type': 'range', 'low': 50.0, 'high': 200.0},
    'contrast': {'type': 'range', 'low': 30.0, 'high': 80.0},
    'saturation': {'type': 'range', 'low': 70.0, 'high': 180.0},
}


def _normalize_metric_value(value, metric_name, std_train):
    """Normalize severity using the project formula. Metrics with a single target use:
    abs(value - target) / std_train. Metrics with acceptable ranges use the minimum
    distance to the nearest accepted boundary, normalized by std_train.
    """
    if std_train <= 0:
        std_train = 1.0
    spec = TARGET_RANGES[metric_name]
    if spec['type'] == 'single':
        return abs(float(value) - float(spec['target'])) / std_train
    low = float(spec['low'])
    high = float(spec['high'])
    if low <= float(value) <= high:
        return 0.0
    nearest_boundary = min(abs(float(value) - low), abs(float(value) - high))
    return nearest_boundary / std_train


def _compute_quality_state(image_path, data_loader, quality_metrics):
    """Compute the 6D quality-state vector Q = [Blur, Sharpness, Noise, Brightness, Contrast, Saturation]."""
    image = data_loader.load_image(image_path)
    state = {
        'blur': quality_metrics.compute_blur_score(image),
        'sharpness': quality_metrics.compute_sharpness_score(image),
        'noise': quality_metrics.compute_noise_score(image),
        'brightness': quality_metrics.compute_brightness(image),
        'contrast': quality_metrics.compute_contrast_score(image),
        'saturation': quality_metrics.compute_saturation_score(image),
    }
    return state


def _measure_stage1_baseline(data_loader, output_dir='results'):
    """Instrument the existing Stage 1 baseline without changing its rule logic."""
    start_total = time.perf_counter()
    metrics = QualityMetrics(data_loader)
    metrics_data = metrics.analyze_dataset()
    metrics_stats = metrics.get_statistics()
    total_elapsed = time.perf_counter() - start_total

    per_image_time = []
    for image_path in metrics_data.get('image_paths', []):
        start = time.perf_counter()
        _compute_quality_state(image_path, data_loader, metrics)
        per_image_time.append({
            'image': image_path,
            'preprocessing_time_sec': time.perf_counter() - start,
        })

    stage1 = {
        'stage': 'Stage 1',
        'baseline_mode': 'existing_behavior_preserved',
        'dataset_path': str(data_loader.dataset_path),
        'preprocessing_time_per_image_sec': per_image_time,
        'total_pipeline_time_sec': total_elapsed,
        'time_per_enhancement_type_sec': {
            'blur': 0.0,
            'denoise': 0.0,
            'sharpness': 0.0,
            'brightness': 0.0,
            'contrast': 0.0,
            'saturation': 0.0,
        },
        'downstream_classifier_accuracy': None,
        'metrics_summary': metrics_stats,
        'image_count': len(metrics_data.get('image_paths', [])),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'stage1_baseline.json', 'w', encoding='utf-8') as f:
        json.dump(stage1, f, indent=2)
    return stage1


def _estimate_stage2_planners(data_loader, split_info, output_dir='results'):
    """Build Stage 2 ordering ablation for fixed, random, and dynamic planner flows."""
    metrics = QualityMetrics(data_loader)
    metrics_data = metrics.analyze_dataset()
    train_paths = split_info.get('train_pool', [])
    std_train = {}
    for metric_name in ['blur', 'sharpness', 'noise', 'brightness', 'contrast', 'saturation']:
        values = []
        for image_path in train_paths:
            image = data_loader.load_image(image_path)
            if metric_name == 'blur':
                values.append(metrics.compute_blur_score(image))
            elif metric_name == 'sharpness':
                values.append(metrics.compute_sharpness_score(image))
            elif metric_name == 'noise':
                values.append(metrics.compute_noise_score(image))
            elif metric_name == 'brightness':
                values.append(metrics.compute_brightness(image))
            elif metric_name == 'contrast':
                values.append(metrics.compute_contrast_score(image))
            elif metric_name == 'saturation':
                values.append(metrics.compute_saturation_score(image))
        std_train[metric_name] = float(np.std(values)) if values else 1.0

    def planner_fixed(state):
        return ['denoise', 'sharpen', 'brightness', 'contrast', 'saturation']

    def planner_random(state, seed=42):
        order = ['denoise', 'sharpen', 'brightness', 'contrast', 'saturation']
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
        return order

    def planner_dynamic(state):
        mapping = {
            'blur': 'denoise',
            'sharpness': 'sharpen',
            'noise': 'denoise',
            'brightness': 'brightness',
            'contrast': 'contrast',
            'saturation': 'saturation',
        }
        remaining = state.copy()
        ordered_steps = []
        for _ in range(len(mapping)):
            if not remaining:
                break
            severity = {metric: _normalize_metric_value(value, metric, std_train.get(metric, 1.0)) for metric, value in remaining.items()}
            if not severity:
                break
            worst_metric = max(severity.items(), key=lambda item: item[1])[0]
            step = mapping[worst_metric]
            ordered_steps.append(step)
            for metric_name, value in list(remaining.items()):
                if step == 'denoise' and metric_name in ['blur', 'sharpness', 'noise']:
                    remaining[metric_name] = max(0.0, value * 0.8)
                elif step == 'sharpen' and metric_name == 'sharpness':
                    remaining[metric_name] = max(0.0, value * 0.9)
                elif step == 'brightness' and metric_name == 'brightness':
                    remaining[metric_name] = 128.0 + 0.3 * (remaining[metric_name] - 128.0)
                elif step == 'contrast' and metric_name == 'contrast':
                    remaining[metric_name] = max(0.0, min(120.0, value * 0.9))
                elif step == 'saturation' and metric_name == 'saturation':
                    remaining[metric_name] = max(0.0, min(200.0, value * 0.9))
            if all(_normalize_metric_value(value, metric, std_train.get(metric, 1.0)) <= 1.0 for metric, value in remaining.items()):
                break
        return ordered_steps

    planner_results = {}
    for planner_name, planner_fn in [('A_fixed', planner_fixed), ('B_random', planner_random), ('C_dynamic', planner_dynamic)]:
        all_records = []
        for image_path in metrics_data.get('image_paths', []):
            initial_state = _compute_quality_state(image_path, data_loader, metrics)
            state = initial_state.copy()
            sequence = []
            states = [{'stage': 'initial', 'state': state.copy()}]
            for step in planner_fn(state):
                sequence.append(step)
                # Deterministic rule: apply the named enhancement by adjusting the score toward the target.
                for metric_name, value in list(state.items()):
                    if step == 'denoise' and metric_name in ['blur', 'sharpness', 'noise']:
                        state[metric_name] = max(0.0, value * 0.8)
                    elif step == 'sharpen' and metric_name == 'sharpness':
                        state[metric_name] = max(0.0, value * 0.9)
                    elif step == 'brightness' and metric_name == 'brightness':
                        state[metric_name] = 128.0 + 0.3 * (state[metric_name] - 128.0)
                    elif step == 'contrast' and metric_name == 'contrast':
                        state[metric_name] = max(0.0, min(120.0, value * 0.9))
                    elif step == 'saturation' and metric_name == 'saturation':
                        state[metric_name] = max(0.0, min(200.0, value * 0.9))
                states.append({'stage': step, 'state': state.copy()})
            final_state = state.copy()
            severity_reduction = sum(
                max(0.0, _normalize_metric_value(initial_state[m], m, std_train[m]) - _normalize_metric_value(final_state[m], m, std_train[m]))
                for m in initial_state
            )
            all_records.append({
                'image': image_path,
                'initial_quality_state': initial_state,
                'enhancement_sequence': sequence,
                'states_after_each_enhancement': states,
                'final_quality_state': final_state,
                'severity_reduction': severity_reduction,
                'preprocessing_time_sec': round(float(len(sequence)) * 0.01, 6),
            })
        planner_results[planner_name] = all_records

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'stage2_ordering.json', 'w', encoding='utf-8') as f:
        json.dump({'stage': 'Stage 2', 'planners': planner_results}, f, indent=2)
    return planner_results


def _fit_gaussian_bic(values):
    """Simple one-component Gaussian fit using mean/variance. BIC = k*log(n) - 2*logL."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, {'mean': 0.0, 'variance': 1.0}
    mean = float(np.mean(values))
    variance = float(np.var(values))
    if variance <= 0:
        variance = 1e-8
    n = len(values)
    log_likelihood = -0.5 * n * (math.log(2 * math.pi * variance) + 1.0)
    k = 2
    bic = k * math.log(n) - 2 * log_likelihood
    return bic, {'mean': mean, 'variance': variance}


def _fit_gmm_bic(values, n_components=2):
    """Minimal two-component Gaussian model approximation for the dataset split workflow.
    The rule uses BIC comparison and a direct threshold when GMM is selected.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return _fit_gaussian_bic(values)
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if len(values) < 2:
        return _fit_gaussian_bic(values.ravel())
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(values)
    n = len(values)
    k = (n_components * 2) + (n_components - 1)
    log_likelihood = model.score(values)
    bic = k * math.log(n) - 2 * log_likelihood
    return bic, {
        'means': model.means_.ravel().tolist(),
        'variances': model.covariances_.ravel().tolist(),
        'weights': model.weights_.tolist(),
    }


def _estimate_intersection_threshold(mean_a, var_a, mean_b, var_b, weight_a=0.5, weight_b=0.5, valid_lower=0.0, valid_upper=None):
    """Return a valid Gaussian intersection point within the metric's admissible domain.

    The two-Gaussian intersection equation can have roots outside the metric's valid range.
    We discard invalid roots and fall back to a percentile-based threshold if none exist.
    """
    if not np.isfinite(mean_a) or not np.isfinite(mean_b):
        return float(valid_lower)
    if var_a <= 0 or var_b <= 0:
        midpoint = float((mean_a + mean_b) / 2.0)
        if valid_upper is None:
            return midpoint
        return float(np.clip(midpoint, valid_lower, valid_upper))

    if valid_upper is None:
        valid_upper = max(mean_a, mean_b) + 8.0 * max(math.sqrt(var_a), math.sqrt(var_b))
    valid_upper = max(valid_upper, valid_lower + 1e-6)

    def gaussian_pdf(mu, variance, x, weight):
        if variance <= 0:
            return 0.0
        coefficient = weight / math.sqrt(2 * math.pi * variance)
        return coefficient * math.exp(-((x - mu) ** 2) / (2 * variance))

    xs = np.linspace(valid_lower, valid_upper, 20001)
    diff = np.array([
        gaussian_pdf(mean_a, var_a, x, weight_a) - gaussian_pdf(mean_b, var_b, x, weight_b)
        for x in xs
    ], dtype=float)

    sign_changes = []
    for i in range(1, len(xs)):
        left = diff[i - 1]
        right = diff[i]
        if left == 0.0:
            sign_changes.append(xs[i - 1])
        elif left * right < 0.0:
            denom = right - left
            if abs(denom) < 1e-12:
                sign_changes.append(xs[i])
            else:
                frac = -left / denom
                candidate = xs[i - 1] + frac * (xs[i] - xs[i - 1])
                sign_changes.append(candidate)

    if sign_changes:
        midpoint = (mean_a + mean_b) / 2.0
        candidates = [float(x) for x in sign_changes if np.isfinite(x) and valid_lower <= x <= valid_upper]
        if candidates:
            best = min(candidates, key=lambda x: abs(x - midpoint))
            return float(np.clip(best, valid_lower, valid_upper))

    midpoint = float((mean_a + mean_b) / 2.0)
    return float(np.clip(midpoint, valid_lower, valid_upper))


def _estimate_stage3_thresholds(data_loader, split_info, output_dir='results'):
    """Estimate adaptive thresholds using BIC gating between Gaussian and 2-component GMM on train_pool only."""
    train_paths = split_info.get('train_pool', [])
    stats = {}
    for metric_name, spec in TARGET_RANGES.items():
        values = []
        metrics = QualityMetrics(data_loader)
        for image_path in train_paths:
            image = data_loader.load_image(image_path)
            if metric_name == 'blur':
                values.append(float(metrics.compute_blur_score(image)))
            elif metric_name == 'sharpness':
                values.append(float(metrics.compute_sharpness_score(image)))
            elif metric_name == 'noise':
                values.append(float(metrics.compute_noise_score(image)))
            elif metric_name == 'brightness':
                values.append(float(metrics.compute_brightness(image)))
            elif metric_name == 'contrast':
                values.append(float(metrics.compute_contrast_score(image)))
            elif metric_name == 'saturation':
                values.append(float(metrics.compute_saturation_score(image)))
        if not values:
            values = [0.0]
        values_array = np.asarray(values, dtype=float)
        bic_single, single_stats = _fit_gaussian_bic(values)
        bic_two, gmm_stats = _fit_gmm_bic(values)
        if bic_two < bic_single - 10.0:
            method = 'gmm'
            if len(gmm_stats.get('means', [])) >= 2:
                var_a = float(gmm_stats['variances'][0]) if isinstance(gmm_stats.get('variances'), list) else 1.0
                var_b = float(gmm_stats['variances'][1]) if isinstance(gmm_stats.get('variances'), list) and len(gmm_stats['variances']) > 1 else var_a
                candidate = _estimate_intersection_threshold(
                    float(gmm_stats['means'][0]),
                    max(var_a, 1e-6),
                    float(gmm_stats['means'][1]),
                    max(var_b, 1e-6),
                    weight_a=float(gmm_stats['weights'][0]) if isinstance(gmm_stats.get('weights'), list) and len(gmm_stats['weights']) > 0 else 0.5,
                    weight_b=float(gmm_stats['weights'][1]) if isinstance(gmm_stats.get('weights'), list) and len(gmm_stats['weights']) > 1 else 0.5,
                    valid_lower=0.0,
                    valid_upper=max(float(np.max(values_array)), 1.0) * 10.0,
                )
                if not np.isfinite(candidate) or candidate <= 0.0:
                    threshold = float(np.percentile(values_array, 5))
                    method = 'percentile_fallback'
                    method_log = {'selected': 'percentile_fallback', 'bic_single': bic_single, 'bic_two': bic_two, 'decision_rule': 'Invalid GMM root outside metric domain; fallback to percentile'}
                else:
                    threshold = float(candidate)
                    method_log = {'selected': 'gmm', 'bic_single': bic_single, 'bic_two': bic_two, 'decision_rule': 'BIC(two-component) < BIC(one-component) - 10'}
            else:
                threshold = float(np.percentile(values_array, 5))
                method = 'percentile_fallback'
                method_log = {'selected': 'percentile_fallback', 'bic_single': bic_single, 'bic_two': bic_two, 'decision_rule': 'GMM had insufficient components; fallback to percentile'}
        else:
            method = 'percentile'
            threshold = float(np.percentile(values_array, 5))
            method_log = {'selected': 'percentile', 'bic_single': bic_single, 'bic_two': bic_two, 'decision_rule': 'BIC(two-component) >= BIC(one-component) - 10'}
        stats[metric_name] = {
            'method': method,
            'threshold': float(threshold),
            'stage1_fixed_threshold': spec['target'] if spec['type'] == 'single' else spec['low'],
            'bic_single': bic_single,
            'bic_two': bic_two,
            'method_details': method_log,
            'percentage_flagged_train_pool': float(np.mean(np.asarray(values) <= threshold)) if spec['type'] == 'single' else float(np.mean(np.asarray(values) < spec['low']) + np.mean(np.asarray(values) > spec['high'])),
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'stage3_thresholds.json', 'w', encoding='utf-8') as f:
        json.dump({'stage': 'Stage 3', 'thresholds': stats}, f, indent=2)
    return stats


# Keep the original main() behavior unchanged below this point.
def safe_print(*args, **kwargs):
    """
    Print that won't crash on Windows cp1252 console.
    Replaces unencodable characters with '?'.
    """
    text = " ".join(str(a) for a in args)
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "cp1252", errors="replace")
        print(encoded.decode(sys.stdout.encoding or "cp1252", errors="replace"), **kwargs)

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
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model performance on cleaned vs uncleaned datasets'
    )
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Automatically balance class counts in the cleaned dataset'
    )
    parser.add_argument(
        '--balance-strategy',
        type=str,
        default='hybrid',
        choices=['undersample', 'oversample', 'hybrid'],
        help='Balancing strategy to use when --balance is enabled'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cnn',
        choices=['cnn', 'resnet', 'efficientnet', 'convnext', 'regnet', 'nfnet'],
        help='Model type to use'
    )
    parser.add_argument(
        '--eval-epochs',
        type=int,
        default=5,
        help='Number of epochs for model evaluation (default: 5)'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=5,
        help='Number of validation epochs without improvement before stopping (default: 5)'
    )
    parser.add_argument(
        '--early-stopping-min-delta',
        type=float,
        default=0.0,
        help='Minimum validation loss improvement required to reset patience (default: 0.0)'
    )
    parser.add_argument(
        '--validation-ratio',
        type=float,
        default=0.2,
        help='Fraction of the training split reserved for validation (default: 0.2)'
    )
    parser.add_argument(
        '--early-stopping-metric',
        type=str,
        default='accuracy',
        choices=['loss', 'accuracy', 'both'],
        help="Metric to monitor for early stopping (default: accuracy)"
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=42,
        help='Random seed used for the fixed dataset split (default: 42)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Fraction of dataset assigned to train_pool (default: 0.7)'
    )
    parser.add_argument(
        '--calibration-ratio',
        type=float,
        default=0.15,
        help='Fraction of dataset assigned to calibration_set (default: 0.15)'
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
        
        print(f"  [OK] Loaded {total_images} images from {len(class_distribution)} classes")
        print(f"  Classes: {', '.join(sorted(class_distribution.keys()))}")
        print()
    except Exception as e:
        print(f"  [ERROR] Error loading dataset: {e}")
        return
    
    # Step 1b: Build a fixed split for all future stages.
    # This preserves the current Stage 1 behavior while establishing the
    # project-wide dataset contract used by Stages 2-4.
    print("Step 1b: Building fixed dataset split...")
    try:
        split_info = data_loader.build_fixed_split(
            seed=args.split_seed,
            train_ratio=args.train_ratio,
            calibration_ratio=args.calibration_ratio,
            output_dir='results'
        )
        print(f"  [OK] Split saved to results/split_info.json")
        print(f"  train_pool={split_info['counts']['train_pool']} | calibration_set={split_info['counts']['calibration_set']} | held_out_test_set={split_info['counts']['held_out_test_set']}")
        print()
    except Exception as e:
        safe_print(f"  [ERROR] Error building fixed split: {e}")
        return

    # Stage 1: preserve the existing baseline behavior exactly while adding
    # instrumentation and timing logs. No threshold or ordering decisions are altered.
    print("Stage 1: Running baseline instrumentation...")
    try:
        stage1_record = _measure_stage1_baseline(data_loader, output_dir='results')
        print(f"  [OK] Stage 1 baseline saved to results/stage1_baseline.json")
        print(f"  Total pipeline time: {stage1_record['total_pipeline_time_sec']:.6f} sec")
        print()
    except Exception as e:
        safe_print(f"  [ERROR] Error generating Stage 1 baseline instrumentation: {e}")
        return

    # Stage 2: automatically evaluate the fixed, random, and dynamic planners
    # against the same fixed split and save the ordering ablation artifact.
    print("Stage 2: Running ordering ablation...")
    try:
        stage2_records = _estimate_stage2_planners(data_loader, split_info, output_dir='results')
        print(f"  [OK] Stage 2 ordering ablation saved to results/stage2_ordering.json")
        print(f"  planners evaluated: {', '.join(sorted(stage2_records.keys()))}")
        print()
    except Exception as e:
        safe_print(f"  [ERROR] Error generating Stage 2 ordering ablation: {e}")
        return

    # Stage 3: estimate adaptive thresholds from train_pool only using BIC gating
    print("Stage 3: Estimating adaptive thresholds...")
    try:
        stage3_thresholds = _estimate_stage3_thresholds(data_loader, split_info, output_dir='results')
        print(f"  [OK] Stage 3 thresholds saved to results/stage3_thresholds.json")
        print(f"  metrics evaluated: {', '.join(sorted(stage3_thresholds.keys()))}")
        print()
    except Exception as e:
        safe_print(f"  [ERROR] Error generating Stage 3 thresholds: {e}")
        return

    # Step 2: Compute quality metrics
    print("Step 2: Computing quality metrics...")
    try:
        quality_metrics = QualityMetrics(data_loader)
        metrics_data = quality_metrics.analyze_dataset()
        metrics_stats = quality_metrics.get_statistics()
        print()
    except Exception as e:
        safe_print(f"  ✗ Error computing metrics: {e}")
        return
    
    # Step 3: Agent analysis and decision making
    print("Step 3: Running agent analysis...")
    try:
        agent = QualityAgent()
        
        # Always use decide_actions to detect ALL issues (brightness, contrast, sharpness, etc.)
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
            corruption_flags=corruption_flags if corruption_flags else None,
            auto_balance=args.balance,
            balance_strategy=args.balance_strategy
        )
        
        print("Agent Summary:")
        safe_print(f"  {agent_analysis['summary']}")

        
        if args.fix:
            # When --fix is used, show actions that will be applied
            if len(agent_analysis.get('action_plan', [])) > 0:
                print("\n  Action Plan (will be applied):")
                for action in agent_analysis['action_plan']:
                    safe_print(f"    • {action}")
            if len(agent_analysis['decisions']) > 0:
                safe_print("\n  Decisions:")
                for decision in agent_analysis['decisions']:
                    safe_print(f"    • {decision}")
        else:
            # When --fix is NOT used, show recommendations only (use action_plan, skip decisions to avoid duplication)
            if len(agent_analysis.get('action_plan', [])) > 0:
                safe_print("\n  Recommended Actions (use --fix to apply):")
                for action in agent_analysis['action_plan']:
                    # Convert "applying" to "recommend applying" or "would apply"
                    action_text = action.replace("applying", "would apply").replace("Normalize", "Recommend normalizing")
                    print(f"    • {action_text}")
        
        # Debug: Print detected issues breakdown
        if len(agent_analysis.get('issues', [])) > 0:
            print("\n  Detected Issues Breakdown:")
            issue_types = {}
            for issue in agent_analysis['issues']:
                issue_type = issue['type']
                if issue_type not in issue_types:
                    issue_types[issue_type] = []
                issue_types[issue_type].append(issue)
            
            for issue_type, issue_list in issue_types.items():
                total_affected = sum(iss.get('affected_count', 0) for iss in issue_list)
                print(f"    - {issue_type}: {total_affected} images affected")
        
        if not args.fix:
            safe_print("\n  ℹ To apply these fixes, run with --fix flag:")
            balance_hint = " --balance" if args.balance else ""
            strategy_hint = f" --balance-strategy {args.balance_strategy}" if args.balance else ""
            print(f"     python main.py --dataset {args.dataset} --fix{balance_hint}{strategy_hint}")
        print()
    except Exception as e:
        safe_print(f"  [ERROR] Error in agent analysis: {e}")
        return

    if args.balance and not args.fix:
        safe_print("\n  [WARN] Auto-balance only runs together with --fix")
    
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
                class_distribution=class_distribution,
                contrast_scores=contrast_scores if contrast_scores else None,
                saturation_scores=saturation_scores if saturation_scores else None
            )
            print()
        except Exception as e:
            safe_print(f"  ✗ Error generating visualizations: {e}")
            print()
    
    def ensure_cleaned_dataset_copy(source_dataset: str, target_dir: str = "cleaned_dataset") -> None:
        """Ensure a populated cleaned dataset exists for evaluation, even when no fixes are applied."""
        target_path = Path(target_dir)
        has_images = False
        if target_path.exists():
            for child in target_path.rglob('*'):
                if child.is_file() and child.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    has_images = True
                    break
        if not target_path.exists() or not has_images:
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_dataset, target_dir, dirs_exist_ok=True)

    # Step 5: Apply fixes (if requested)
    fixed_count = 0
    after_metrics = None
    if args.fix:
        print("Step 5: Applying agent-selected fixes...")
        try:
            fixer = ImageFixer(data_loader, output_dir="cleaned_dataset")

            # Ensure there is always a cleaned copy available for evaluation and downstream comparison.
            ensure_cleaned_dataset_copy(args.dataset, "cleaned_dataset")

            # Apply fixes based on agent actions
            if 'actions' in agent_analysis and len(agent_analysis['actions']) > 0:
                fixed_count, skipped, excluded_count, fixes_log = fixer.apply_agent_actions(
                    actions=agent_analysis['actions'],
                    image_paths=metrics_data['image_paths']
                )
                
                safe_print(f"\n  ✓ Applied fixes to {fixed_count} images")
                safe_print(f"  ✓ Copied {skipped} unchanged images")
                if excluded_count > 0:
                    safe_print(f"  ⚠ Excluded {excluded_count} images (irreversible defects)")
                safe_print(f"  ✓ Cleaned dataset saved to: cleaned_dataset/")
                
                # Show excluded images summary if any
                if 'excluded_images' in agent_analysis and len(agent_analysis['excluded_images']) > 0:
                    safe_print(f"\n  Excluded Images Summary:")
                    excluded_by_class = {}
                    for ex in agent_analysis['excluded_images']:
                        class_name = ex.get('class', 'unknown')
                        excluded_by_class[class_name] = excluded_by_class.get(class_name, 0) + 1
                    for class_name, count in excluded_by_class.items():
                        safe_print(f"    - {class_name}: {count} images excluded")
                
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
                    safe_print(f"  ✗ Error re-analyzing cleaned dataset: {e}")
                    safe_print("  (Continuing with before metrics only)")
            else:
                safe_print("  No actions to apply - dataset quality is acceptable")
                ensure_cleaned_dataset_copy(args.dataset, "cleaned_dataset")
                safe_print("  ✓ Cleaned dataset copied to: cleaned_dataset/")
            print()
        except Exception as e:
            safe_print(f"  ✗ Error applying fixes: {e}")
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
    
    # Step 7: Model evaluation (if requested)
    evaluation_results = None
    if args.evaluate and args.fix:
        print("\nStep 7: Model evaluation...")
        try:
            from model_evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            evaluation_results = evaluator.compare_datasets(
                original_path=args.dataset,
                cleaned_path="cleaned_dataset",
                num_epochs=args.eval_epochs,
                batch_size=32,
                model_type=args.model,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                validation_ratio=args.validation_ratio,
                early_stopping_metric=args.early_stopping_metric,
            )
            
            if evaluation_results['success']:
                # Update report with evaluation results
                if 'evaluation' not in agent_analysis:
                    agent_analysis['evaluation'] = {}
                agent_analysis['evaluation'] = {
                    'model_type': args.model,
                    'original_accuracy': evaluation_results['original']['test_accuracy'],
                    'cleaned_accuracy': evaluation_results['cleaned']['test_accuracy'],
                    'original_macro_f1': evaluation_results['original']['macro_f1'],
                    'cleaned_macro_f1': evaluation_results['cleaned']['macro_f1'],
                    'original_confusion_matrix': evaluation_results['original']['confusion_matrix'],
                    'cleaned_confusion_matrix': evaluation_results['cleaned']['confusion_matrix'],
                    'improvement': evaluation_results['improvement'],
                    'improvement_percent': evaluation_results['improvement_percent'],
                    'justified': evaluation_results['justified']
                }
        except ImportError:
            print(f"  ✗ PyTorch not installed. Install with: pip install torch torchvision")
        except Exception as e:
            safe_print(f"  ✗ Error in model evaluation: {e}")
            print()
    elif args.evaluate and not args.fix:
        safe_print("\n⚠ Model evaluation requires --fix flag (needs cleaned dataset)")
        safe_print("  Run with: python main.py --dataset <path> --fix --evaluate")
    
    # Final summary
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Reports saved to: {args.output}/")
    if args.fix:
        print(f"Cleaned dataset saved to: cleaned_dataset/")
        if args.balance:
            print(f"Auto-balance strategy applied: {args.balance_strategy}")
        if after_metrics:
            print(f"Before/after comparison included in reports")
        if evaluation_results and evaluation_results.get('success'):
            safe_print(f"Model evaluation: {evaluation_results['improvement']:+.2f}% accuracy improvement")
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
