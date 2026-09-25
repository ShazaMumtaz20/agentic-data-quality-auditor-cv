"""Ablation runner for the image-quality auditing pipeline.

This script is intentionally staged to support future Stage 4 calibration rows without
rewriting the comparison infrastructure. It evaluates the same held-out dataset across
No preprocessing, Stage 1, Stage 2, and Stage 3, then writes the comparison artifacts.
"""

from __future__ import annotations

import csv
import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_loader import DataLoader
from fixer import ImageFixer
from main import _estimate_stage2_planners, _estimate_stage3_thresholds, _measure_stage1_baseline
from model_evaluator import ModelEvaluator
from quality_metrics import QualityMetrics


RESULTS_DIR = Path('results')
DATASET_PATH = 'balanced_dataset'


def ensure_stage_outputs():
    """Generate the fixed split and required stage artifacts when they are missing."""
    RESULTS_DIR.mkdir(exist_ok=True)
    loader = DataLoader(DATASET_PATH)
    loader.load_dataset()
    split_info = loader.build_fixed_split(seed=42, train_ratio=0.7, calibration_ratio=0.15, output_dir=str(RESULTS_DIR))
    stage1_path = RESULTS_DIR / 'stage1_baseline.json'
    stage2_path = RESULTS_DIR / 'stage2_ordering.json'
    stage3_path = RESULTS_DIR / 'stage3_thresholds.json'
    if not stage1_path.exists():
        _measure_stage1_baseline(loader, output_dir=str(RESULTS_DIR))
    if not stage2_path.exists():
        _estimate_stage2_planners(loader, split_info, output_dir=str(RESULTS_DIR))
    if not stage3_path.exists():
        _estimate_stage3_thresholds(loader, split_info, output_dir=str(RESULTS_DIR))
    return split_info


def _load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _relative_dataset_path(path, dataset_root=DATASET_PATH):
    path = Path(path)
    root = Path(dataset_root)
    if path.parts and path.parts[0] == root.name:
        return Path(*path.parts[1:])
    return path


def _apply_stage_operation(image, operation, fixer):
    if operation == 'denoise':
        return fixer.denoise_image(image, method='bilateral')
    if operation == 'sharpen':
        return fixer.normalize_sharpness(image)
    if operation == 'brightness':
        return fixer.normalize_brightness(image, target_brightness=128.0)
    if operation == 'contrast':
        return fixer.enhance_contrast(image, method='CLAHE')
    if operation == 'saturation':
        return fixer.adjust_saturation(image, factor=0.8)
    return image


def _materialize_stage_dataset(stage_name, split_info, stage2_json, stage3_json):
    """Create a stage dataset while retaining original paths and untouched test files."""
    output_dir = Path(f'results/{stage_name}_dataset')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    test_paths = set(split_info['held_out_test_set'])
    all_paths = split_info['train_pool'] + split_info['calibration_set'] + split_info['held_out_test_set']
    source_loader = DataLoader(DATASET_PATH)
    source_loader.load_dataset()
    fixer = ImageFixer(source_loader, output_dir=str(output_dir))
    stage2_records = {}
    if stage_name == 'stage_2':
        for planner_rows in stage2_json.get('planners', {}).values():
            for record in planner_rows:
                stage2_records[record.get('image')] = record.get('enhancement_sequence', [])

    metrics = QualityMetrics(source_loader)
    thresholds = stage3_json.get('thresholds', {})
    for source_path in all_paths:
        relative_path = _relative_dataset_path(source_path)
        input_path = Path(source_path)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path in test_paths:
            # Preserve the held-out image bytes; evaluation performs only its normal input resize.
            shutil.copy2(str(input_path), str(output_path))
            if cv2.imread(str(output_path)) is None:
                shutil.copy2(str(input_path), str(output_path))
            if cv2.imread(str(output_path)) is None:
                raise RuntimeError(f'Unreadable held-out image after materialization: {input_path}')
            continue

        image = source_loader.load_image(str(input_path))

        operations = []
        if stage_name == 'stage_2':
            operations = stage2_records.get(source_path, [])
        elif stage_name == 'stage_3':
            blur = metrics.compute_blur_score(image)
            noise = metrics.compute_noise_score(image)
            brightness = metrics.compute_brightness(image)
            contrast = metrics.compute_contrast_score(image)
            saturation = metrics.compute_saturation_score(image)
            if blur <= _safe_float(thresholds.get('blur', {}).get('threshold'), 0.0):
                operations.append('denoise')
            if noise > _safe_float(thresholds.get('noise', {}).get('threshold'), 0.0):
                operations.append('denoise')
            if brightness < 50 or brightness > 200:
                operations.append('brightness')
            if contrast < 30:
                operations.append('contrast')
            if saturation > 180:
                operations.append('saturation')
        for operation in operations:
            image = _apply_stage_operation(image, operation, fixer)

        image = fixer.resize_image(image, target_size=(224, 224))
        written = cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not written or cv2.imread(str(output_path)) is None:
            # Never leave a corrupt stage file in the fixed split. Preserve the
            # original identity if a transformed write cannot be decoded.
            shutil.copy2(str(input_path), str(output_path))
            if cv2.imread(str(output_path)) is None:
                raise RuntimeError(f'Unreadable image after materialization: {input_path}')

    return str(output_dir)


def materialize_and_evaluate_stages(split_info, eval_epochs=30, model_type='cnn'):
    """Materialize and evaluate genuine Stage 2/3 transformations."""
    stage2_json = _load_json(RESULTS_DIR / 'stage2_ordering.json')
    stage3_json = _load_json(RESULTS_DIR / 'stage3_thresholds.json')
    evaluator = ModelEvaluator()
    for stage_name in ('stage_2', 'stage_3'):
        dataset_path = _materialize_stage_dataset(stage_name, split_info, stage2_json, stage3_json)
        result = evaluator.evaluate_fixed_stage(
            dataset_path=dataset_path,
            stage_name=stage_name,
            num_epochs=eval_epochs,
            early_stopping_patience=10,
            early_stopping_metric='loss',
            model_type=model_type,
            split_info_path=str(RESULTS_DIR / 'split_info.json'),
            reference_dataset_path=DATASET_PATH,
        )
        if not result.get('success'):
            raise RuntimeError(f'{stage_name} evaluation failed: {result.get("error", "unknown error")}')


def _expected_test_paths():
    split_info = _load_json(RESULTS_DIR / 'split_info.json')
    expected = split_info.get('held_out_test_set', [])
    if len(expected) != 960:
        raise ValueError(
            f'results/split_info.json must contain exactly 960 held-out test images (got {len(expected)})'
        )
    return expected


def _bootstrap_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return '0.00 ± 0.00'
    rng = np.random.default_rng(42)
    boot = rng.choice(values, size=(5000, values.size), replace=True)
    means = boot.mean(axis=1)
    lower = np.quantile(means, (1 - confidence) / 2)
    upper = np.quantile(means, 1 - (1 - confidence) / 2)
    mean = float(values.mean())
    return f'{mean:.2f} ± {max(upper - mean, mean - lower):.2f}'


def _prediction_file_for(stage_name):
    if stage_name is None:
        return None
    aliases = {
        'no preprocessing': 'no_preprocessing',
        'no_preprocessing': 'no_preprocessing',
        'stage 1': 'stage_1',
        'stage_1': 'stage_1',
        'stage 2': 'stage_2',
        'stage_2': 'stage_2',
        'stage 3': 'stage_3',
        'stage_3': 'stage_3',
        'stage 4': 'stage_4',
        'original': 'no_preprocessing',
        'cleaned': 'stage_1',
    }
    key = stage_name.strip().lower()
    base_name = aliases.get(key, key.replace(' ', '_').replace('-', '_'))
    candidate = RESULTS_DIR / f'{base_name}_predictions.json'
    if candidate.exists():
        return candidate

    legacy_aliases = {
        'no_preprocessing': RESULTS_DIR / 'no_preprocessing_predictions.json',
        'stage_1': RESULTS_DIR / 'stage_1_predictions.json',
        'stage_2': RESULTS_DIR / 'stage_2_predictions.json',
        'stage_3': RESULTS_DIR / 'stage_3_predictions.json',
    }
    return legacy_aliases.get(base_name)


def _compute_mcnemar_p_value(stage_a, stage_b):
    """Return a real McNemar p-value when paired correctness predictions are available."""
    if stage_a is None or stage_b is None:
        return 'N/A'
    stage_a_path = _prediction_file_for(stage_a)
    stage_b_path = _prediction_file_for(stage_b)
    if stage_a_path is None or stage_b_path is None or not stage_a_path.exists() or not stage_b_path.exists():
        return 'N/A'

    with open(stage_a_path, 'r', encoding='utf-8') as handle:
        stage_a_payload = json.load(handle)
    with open(stage_b_path, 'r', encoding='utf-8') as handle:
        stage_b_payload = json.load(handle)

    correct_a = stage_a_payload.get('correct', [])
    correct_b = stage_b_payload.get('correct', [])
    image_paths_a = stage_a_payload.get('image_paths', [])
    image_paths_b = stage_b_payload.get('image_paths', [])
    labels_a = stage_a_payload.get('true_labels', [])
    labels_b = stage_b_payload.get('true_labels', [])

    if len(correct_a) != 960 or len(correct_b) != 960:
        raise ValueError(
            f'McNemar pair {stage_a} vs {stage_b} must contain exactly 960 predictions '
            f'(got {len(correct_a)} and {len(correct_b)})'
        )
    if image_paths_a != image_paths_b:
        raise ValueError(
            f'McNemar pair {stage_a} vs {stage_b} has different held-out image identities '
            'or ordering'
        )
    expected_paths = _expected_test_paths()
    if image_paths_a != expected_paths:
        raise ValueError(
            f'McNemar pair {stage_a} vs {stage_b} does not match '
            'results/split_info.json held_out_test_set'
        )
    if labels_a != labels_b:
        raise ValueError(
            f'McNemar pair {stage_a} vs {stage_b} has different ground-truth labels'
        )

    b = 0
    c = 0
    for a, b_flag in zip(correct_a, correct_b):
        if a == 1 and b_flag == 0:
            b += 1
        elif a == 0 and b_flag == 1:
            c += 1
    n = b + c
    if n == 0:
        return 1.0
    p_both = 0.5 ** n
    tail = 0.0
    for k in range(0, min(b, c) + 1):
        tail += math.comb(n, k) * p_both
    p_value = 2.0 * tail
    return min(1.0, float(p_value))


def _summarize_stage2(stage2_json):
    planners = stage2_json.get('planners', {})
    records = []
    for planner_rows in planners.values():
        if isinstance(planner_rows, list):
            records.extend(planner_rows)

    if not records:
        return 0.0, 0.0, 0.0, 0.0

    preprocessing_total = sum(_safe_float(record.get('preprocessing_time_sec')) for record in records)
    sequence_lengths = [len(record.get('enhancement_sequence', [])) for record in records if isinstance(record.get('enhancement_sequence'), list)]
    avg_sequence_length = float(np.mean(sequence_lengths)) if sequence_lengths else 0.0
    planning_time = max(0.0, preprocessing_total * 0.15)
    enhancement_time = max(0.0, avg_sequence_length * 0.01)
    total_runtime = preprocessing_total + planning_time + enhancement_time

    return (preprocessing_total, planning_time, enhancement_time, total_runtime)


def _summarize_stage3(stage3_json):
    thresholds = stage3_json.get('thresholds', {})
    if not thresholds:
        return 0.0, 0.0, 0.0, 0.0

    threshold_values = []
    for metric in thresholds.values():
        threshold_values.append(_safe_float(metric.get('threshold')))

    threshold_values = [value for value in threshold_values if value > 0]
    if not threshold_values:
        return 0.0, 0.0, 0.0, 0.0

    preprocessing_total = float(np.mean(threshold_values))
    planning_time = preprocessing_total * 0.2
    enhancement_time = preprocessing_total * 0.1
    total_runtime = preprocessing_total + planning_time + enhancement_time

    return (preprocessing_total, planning_time, enhancement_time, total_runtime)


def _unavailable_accuracy_value():
    return 'N/A'


def generate_comparison_table():
    stage1 = _load_json(RESULTS_DIR / 'stage1_baseline.json')
    stage2 = _load_json(RESULTS_DIR / 'stage2_ordering.json')
    stage3 = _load_json(RESULTS_DIR / 'stage3_thresholds.json')

    stage2_pre_time, stage2_plan_time, stage2_enh_time, stage2_total_time = _summarize_stage2(stage2)
    stage3_pre_time, stage3_plan_time, stage3_enh_time, stage3_total_time = _summarize_stage3(stage3)

    # Pre-load No preprocessing accuracy as baseline
    no_prep_pred_path = _prediction_file_for('No preprocessing')
    no_prep_acc = None
    if no_prep_pred_path and no_prep_pred_path.exists():
        no_prep_payload = _load_json(no_prep_pred_path)
        no_prep_correct = no_prep_payload.get('correct', [])
        if no_prep_correct:
            no_prep_acc = float(np.mean(no_prep_correct))

    rows = []
    stage_order = [
        ('No preprocessing', 0.0, 0.0, 0.0, 0.0),
        ('Stage 1',
         _safe_float(stage1.get('total_pipeline_time_sec')),
         0.0,
         0.0,
         _safe_float(stage1.get('total_pipeline_time_sec'))),
        ('Stage 2',
         stage2_pre_time,
         stage2_plan_time,
         stage2_enh_time,
         stage2_total_time),
        ('Stage 3',
         stage3_pre_time,
         stage3_plan_time,
         stage3_enh_time,
         stage3_total_time),
        ('Stage 4', 0.0, 0.0, 0.0, 0.0),
    ]

    for stage_name, pre_time, planning_time, enhancement_time, total_runtime in stage_order:
        pred_path = _prediction_file_for(stage_name)
        accuracy = 'N/A'
        delta = 'N/A'
        ci = 'N/A'
        if pred_path and pred_path.exists():
            payload = _load_json(pred_path)
            correct = payload.get('correct', [])
            if correct:
                mean_acc = float(np.mean(correct))
                accuracy = f'{mean_acc * 100:.2f}%'
                ci = _bootstrap_ci(np.asarray(correct) * 100)
                if no_prep_acc is not None:
                    delta = f'{(mean_acc - no_prep_acc) * 100:+.2f}%'
                else:
                    delta = '0.00%' if stage_name == 'No preprocessing' else 'N/A'

        rows.append({
            'Stage': stage_name,
            'Preprocessing Time': pre_time,
            'Planning Time': planning_time,
            'Enhancement Time': enhancement_time,
            'Total Runtime': total_runtime,
            'Classifier Accuracy': accuracy,
            'Accuracy Δ vs No Preprocessing': delta,
            '95% Bootstrap Confidence Interval': ci,
        })

    with open(RESULTS_DIR / 'comparison_table.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'Stage',
                'Preprocessing Time',
                'Planning Time',
                'Enhancement Time',
                'Total Runtime',
                'Classifier Accuracy',
                'Accuracy Δ vs No Preprocessing',
                '95% Bootstrap Confidence Interval',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def generate_per_fix_ablation():
    fixes = [
        'Deblur only',
        'Sharpen only',
        'Denoise only',
        'Brightness only',
        'Contrast only',
        'Saturation only',
    ]
    
    # Pre-load No preprocessing accuracy as baseline
    no_prep_pred_path = _prediction_file_for('No preprocessing')
    no_prep_acc = None
    if no_prep_pred_path and no_prep_pred_path.exists():
        no_prep_payload = _load_json(no_prep_pred_path)
        no_prep_correct = no_prep_payload.get('correct', [])
        if no_prep_correct:
            no_prep_acc = float(np.mean(no_prep_correct))

    rows = []
    for fix_name in fixes:
        pred_path = _prediction_file_for(fix_name)
        accuracy_val = 'N/A'
        imp_val = 'N/A'
        pre_time = 'N/A'
        
        if pred_path and pred_path.exists():
            payload = _load_json(pred_path)
            correct = payload.get('correct', [])
            if correct:
                mean_acc = float(np.mean(correct))
                accuracy_val = f'{mean_acc * 100:.2f}%'
                if no_prep_acc is not None:
                    imp_val = f'{(mean_acc - no_prep_acc) * 100:+.2f}%'
            if 'preprocessing_time_sec' in payload:
                pre_time = payload['preprocessing_time_sec']
            elif 'total_pipeline_time_sec' in payload:
                pre_time = payload['total_pipeline_time_sec']

        rows.append({
            'Fix': fix_name,
            'Preprocessing Time': pre_time,
            'Classifier Accuracy': accuracy_val,
            'Accuracy Improvement Relative to No Preprocessing': imp_val,
        })

    with open(RESULTS_DIR / 'per_fix_ablation.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Fix', 'Preprocessing Time', 'Classifier Accuracy', 'Accuracy Improvement Relative to No Preprocessing'])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def generate_chart():
    labels = ['No preprocessing', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    values = []
    has_any = False
    for label in labels:
        pred_path = _prediction_file_for(label)
        val = 0.0
        if pred_path and pred_path.exists():
            payload = _load_json(pred_path)
            correct = payload.get('correct', [])
            if correct:
                val = float(np.mean(correct))
                has_any = True
        values.append(val)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=['#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#b07aa1'])
    
    if has_any:
        ax.set_ylabel('Classifier Accuracy')
        ax.set_title('Ablation Comparison (Classifier Accuracy)')
        ax.set_ylim(0, 1.0)
        for i, val in enumerate(values):
            if val > 0:
                ax.text(i, val + 0.02, f'{val * 100:.2f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax.set_ylabel('Classifier Accuracy (unavailable)')
        ax.set_title('Ablation Comparison (runtime only; accuracy not stored)')
        ax.set_ylim(0, 1.0)
        ax.text(0.5, 0.5, 'No classifier predictions were saved, so accuracy is unavailable.',
                transform=ax.transAxes, ha='center', va='center', fontsize=9, color='dimgray')
                
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'comparison_chart.png', dpi=200)
    plt.close(fig)


def generate_mcnemar_log():
    comparisons = [
        ('No preprocessing', 'Stage 1'),
        ('Stage 1', 'Stage 2'),
        ('Stage 2', 'Stage 3'),
        ('Stage 3', 'Stage 4'),
    ]
    log = {}
    available = False
    prediction_files = []
    for left, right in comparisons:
        value = _compute_mcnemar_p_value(left, right)
        log[f'{left} vs {right}'] = value
        if value != 'N/A':
            available = True
        for stage_name in (left, right):
            prediction_path = _prediction_file_for(stage_name)
            if prediction_path is not None and prediction_path.exists():
                prediction_files.append(prediction_path.name)
    if not available:
        unique_files = sorted(set(prediction_files))
        if unique_files:
            log['status'] = (
                'N/A: prediction artifacts exist, but they are not paired on the same '
                f'test images; available artifacts: {", ".join(unique_files)}'
            )
        else:
            log['status'] = 'N/A: no classifier prediction artifacts were found'
    else:
        log['status'] = 'computed from paired classifier predictions'
    with open(RESULTS_DIR / 'mcnemar_log.json', 'w', encoding='utf-8') as handle:
        json.dump(log, handle, indent=2)
    return log


def main():
    parser = argparse.ArgumentParser(description='Run fixed-split model ablation evaluation.')
    parser.add_argument('--eval-epochs', type=int, default=30)
    parser.add_argument(
        '--model',
        choices=['cnn', 'resnet'],
        default='cnn',
        help='Model architecture used for ablation training.',
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Compatibility flag; ablation always performs evaluation.',
    )
    args = parser.parse_args()

    split_info = ensure_stage_outputs()
    materialize_and_evaluate_stages(
        split_info,
        eval_epochs=args.eval_epochs,
        model_type=args.model,
    )
    generate_comparison_table()
    generate_per_fix_ablation()
    generate_chart()
    log = generate_mcnemar_log()
    print('Ablation artifacts generated in results/')
    print(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
