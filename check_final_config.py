import torch

# 最終モデルの設定を確認
checkpoint = torch.load('results/task3_training_final/models/final_model.pth', map_location='cpu')
config = checkpoint['config']
df_state = config['ssm']['df_state']

print('=== Final Model Config Check ===')
print('DF State keys:', list(df_state.keys()))

if 'feature_net' in df_state:
    print('✅ SUCCESS: feature_net =', df_state['feature_net'])
else:
    print('❌ FAILED: No feature_net')

# 完全な設定も表示
print('\n=== Complete DF State Config ===')
for key, value in df_state.items():
    print(f'{key}: {value}')