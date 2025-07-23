from model import Unet, GaussianDiffusion, Trainer
import torch
import sys
import os
from pathlib import Path

def find_latest_checkpoint(results_folder='./results'):
    """Find the latest checkpoint file in the results folder"""
    results_path = Path(results_folder)
    if not results_path.exists():
        return None
    
    # Look for model files with pattern model-{milestone}.pt
    model_files = list(results_path.glob('model-*.pt'))
    if not model_files:
        return None
    
    # Extract milestone numbers and find the latest
    milestones = []
    for file in model_files:
        try:
            # Extract number from filename like "model-5.pt"
            milestone = int(file.stem.split('-')[1])
            milestones.append(milestone)
        except (ValueError, IndexError):
            continue
    
    if milestones:
        return max(milestones)
    return None

def main():
    # Check for command line argument
    milestone = None
    if len(sys.argv) > 1:
        try:
            milestone = int(sys.argv[1])
            print(f"Loading from specified checkpoint: model-{milestone}.pt")
        except ValueError:
            print("Error: Milestone must be a number")
            sys.exit(1)
    else:
        # Find latest checkpoint automatically
        milestone = find_latest_checkpoint()
        if milestone is None:
            print("No checkpoints found. Starting fresh training.")
            print("Use train.py for fresh training or provide a milestone number.")
            sys.exit(1)
        else:
            print(f"Found latest checkpoint: model-{milestone}.pt")

    print(milestone)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4),
        channels=3,
        self_condition=True,
        learned_variance=False,
        flash_attn=False,
    )

    # 2. Wrap with GaussianDiffusion (same as train.py)
    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_v',
        beta_schedule='sigmoid',
        auto_normalize=True
    )

    # 3. Define trainer (same as train.py)
    trainer = Trainer(
        diffusion_model=diffusion,
        folder='./archive',
        train_batch_size=16,
        gradient_accumulate_every=2,
        train_lr=1e-4,
        train_num_steps=80000,
        ema_update_every=10,
        ema_decay=0.995,
        save_and_sample_every=1000,
        num_samples=36,
        results_folder='./results',
        amp=True,
        calculate_fid=False,  # disabled to avoid numerical errors
        save_best_and_latest_only=False,
    )

    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

    # Load checkpoint
    print(f"Loading checkpoint from model-{milestone}.pt...")
    trainer.load(milestone)
    print(f"Resumed training from step {trainer.step}")

    # Continue training
    trainer.train()

if __name__ == "__main__":
    main()
