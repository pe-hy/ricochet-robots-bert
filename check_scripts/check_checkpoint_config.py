"""Check the configuration stored in the checkpoint"""
import torch

checkpoint_path = "./tmp/checkpoints/d256_l24_h16_n50000_lr0.001_bs256/epoch=45-0.8828.ckpt"

print("="*80)
print("LOADING CHECKPOINT")
print("="*80)
print(f"Path: {checkpoint_path}\n")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  {key}")

print(f"\n" + "="*80)
print("HYPERPARAMETERS")
print("="*80)

if 'hyper_parameters' in checkpoint:
    hparams = checkpoint['hyper_parameters']
    for key, value in sorted(hparams.items()):
        print(f"{key}: {value}")
else:
    print("No hyper_parameters found in checkpoint")

# Check model state dict for clues
print(f"\n" + "="*80)
print("MODEL STATE DICT (first 20 keys)")
print("="*80)

if 'state_dict' in checkpoint:
    state_dict_keys = list(checkpoint['state_dict'].keys())
    for key in state_dict_keys[:20]:
        print(f"  {key}")
    print(f"  ... ({len(state_dict_keys)} total keys)")
