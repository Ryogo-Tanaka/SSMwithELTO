import torch

# デバッグモデルの設定を確認
checkpoint = torch.load('results/task3_training_debug/models/final_model.pth', map_location='cpu')

print('=== Checkpoint Keys ===')
print('Keys in checkpoint:', list(checkpoint.keys()))
print()

if 'config' in checkpoint:
    config = checkpoint['config']
    print('=== Config Structure ===')
    print('Config keys:', list(config.keys()))
    
    if 'ssm' in config:
        ssm = config['ssm']
        print('SSM keys:', list(ssm.keys()))
        
        if 'df_state' in ssm:
            df_state = ssm['df_state']
            print('DF State keys:', list(df_state.keys()))
            print('DF State content:', df_state)
            
            # feature_netの有無を確認
            if 'feature_net' in df_state:
                print('✅ FOUND: feature_net =', df_state['feature_net'])
            else:
                print('❌ MISSING: feature_net not found')
        else:
            print('❌ No df_state in ssm')
    else:
        print('❌ No ssm in config')
else:
    print('❌ No config key in checkpoint')