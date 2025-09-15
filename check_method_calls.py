import torch
import sys
sys.path.append('/workspace/nas/SSMwithELTO')

# TwoStageTrainerの_build_complete_configメソッドにDEBUGログが追加されているか確認
from src.training.two_stage_trainer import TwoStageTrainer
import inspect

# _extract_df_state_configメソッドのソースを確認
source = inspect.getsource(TwoStageTrainer._extract_df_state_config)
print('=== _extract_df_state_config Method Source ===')
print(source[:500] + '...' if len(source) > 500 else source)

# DEBUGが含まれているか確認
if 'DEBUG:' in source:
    print('✅ DEBUG logs are present in method')
else:
    print('❌ No DEBUG logs found in method')