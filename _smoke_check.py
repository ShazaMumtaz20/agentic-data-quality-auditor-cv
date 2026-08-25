from data_loader import DataLoader
from main import _estimate_stage2_planners

loader = DataLoader('balanced_dataset')
loader.load_dataset()
split = loader.build_fixed_split(seed=42, train_ratio=0.7, calibration_ratio=0.15, output_dir='results')
planner = _estimate_stage2_planners(loader, split, output_dir='results')
print('dynamic_len', len(planner['C_dynamic'][0]['enhancement_sequence']))
print('dynamic_seq', planner['C_dynamic'][0]['enhancement_sequence'])
print('fixed_seq', planner['A_fixed'][0]['enhancement_sequence'])
print('random_seq', planner['B_random'][0]['enhancement_sequence'])
