import torch

checkpoint = torch.load('results/task3_training_v3/models/final_model.pth', map_location='cpu')
config = checkpoint['config']

print('=== Complete Config Structure ===')
print('Top level keys:', list(config.keys()))
print()

print('=== SSM Config ===')
ssm_config = config['ssm']
print('SSM keys:', list(ssm_config.keys()))
print()

print('=== DF State Config ===')
df_state = ssm_config['df_state']
print('DF State keys:', list(df_state.keys()))
print('DF State content:', df_state)
print()

print('=== DF Observation Config ===')
df_obs = ssm_config['df_observation']
print('DF Obs keys:', list(df_obs.keys()))
print('DF Obs content:', df_obs)