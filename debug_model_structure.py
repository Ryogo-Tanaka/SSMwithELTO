import torch
cp = torch.load('results/task3_debug9/models/final_model.pth', map_location='cpu')
encoder_dict = cp['model_state_dict']['encoder']

print('=== ENCODER PARAMETER SHAPES ===')
for key, tensor in encoder_dict.items():
    print(f'{key}: {tensor.shape}')

print('\n=== CHANNELS DETECTION ===')
# in_proj.weightからchannels値を検出
if 'in_proj.weight' in encoder_dict:
    in_proj_shape = encoder_dict['in_proj.weight'].shape
    channels = in_proj_shape[0]  # [channels, input_dim, kernel_size]
    print(f'Detected channels: {channels}')

# tcn layersの数も検出
tcn_layers = []
for key in encoder_dict.keys():
    if key.startswith('tcn.') and '.conv.weight' in key:
        layer_num = int(key.split('.')[1])
        tcn_layers.append(layer_num)

if tcn_layers:
    max_layer = max(tcn_layers)
    total_layers = max_layer + 1
    print(f'Detected layers: {total_layers}')