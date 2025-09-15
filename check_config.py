import torch

checkpoint = torch.load('results/task3_training_v3/models/final_model.pth', map_location='cpu')
config = checkpoint['config']

print('=== DF State Config ===')
df_state = config['ssm']['df_state']
print('Keys:', list(df_state.keys()))

if 'feature_net' in df_state:
    print('✅ Feature Net Config:', df_state['feature_net'])
else:
    print('❌ No feature_net found!')