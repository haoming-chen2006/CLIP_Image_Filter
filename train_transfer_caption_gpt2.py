import os
import random
import time
import glob
from typing import List, Dict, Any

import cv2
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from transformers import AutoTokenizer

from modules import TransferHead, GPT
from transformers import DistilBertTokenizer
from config import TransferGPT2Config as CFG
from dataset import CLIPDataset, get_transforms, load_flickr_data
from clip import CLIPModel
from utils import AvgMeter, get_lr

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


class CLIPTransferCaptionModelGPT2(nn.Module):
    """Generate captions using frozen CLIP image encoder and GPT2 from modules."""

    def __init__(self, gpt_name: str = "gpt2"):
        super().__init__()
        # Load CLIP and freeze it
        self.clip = CLIPModel()
        
        # Construct path to the trained CLIP model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        clip_model_path = os.path.join(script_dir, "best.pt")

        if not os.path.exists(clip_model_path):
            raise FileNotFoundError(f"Trained CLIP model not found at {clip_model_path}. Please run train.py first.")

        # Load the state dict from your trained model
        self.clip.load_state_dict(torch.load(clip_model_path, map_location=CFG.device))
        
        for p in self.clip.parameters():
            p.requires_grad = False

        # Load GPT2 from modules with pretrained weights
        self.lm = GPT.from_pretrained(gpt_name)

        # Projection from CLIP embedding (256) to GPT2 hidden dim
        self.transfer_head = TransferHead(CFG.projection_dim, self.lm.config.n_embd)

    def forward(self, batch):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.no_grad():
            img_feat = self.clip.image_encoder(images)
            clip_embed = self.clip.image_projection(img_feat)
        prefix = self.transfer_head(clip_embed).unsqueeze(1)

        token_embeds = self.lm.transformer.wte(input_ids)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        b, t, _ = inputs_embeds.size()
        pos = torch.arange(0, t, dtype=torch.long, device=inputs_embeds.device)
        pos_emb = self.lm.transformer.wpe(pos)
        x = self.lm.transformer.drop(inputs_embeds + pos_emb)
        for block in self.lm.transformer.h:
            x = block(x)
        x = self.lm.transformer.ln_f(x)
        logits = self.lm.lm_head(x)
        # The prefix is not part of the target, so we shift the logits and labels
        logits = logits[:, :-1, :]
        targets = input_ids

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return loss



def split_data(image_names, captions, train_ratio=0.8, val_ratio=0.1):
    data = list(zip(image_names, captions))
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    train_images, train_captions = zip(*train_data)
    val_images, val_captions = zip(*val_data)
    return list(train_images), list(train_captions), list(val_images), list(val_captions)


def build_loaders(train_images, train_captions, val_images, val_captions, tokenizer, ddp=False):
    train_dataset = CLIPDataset(train_images, train_captions, tokenizer, get_transforms("train"))
    val_dataset = CLIPDataset(val_images, val_captions, tokenizer, get_transforms("valid"))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if ddp else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if ddp else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
        sampler=val_sampler,
    )
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, device, master_process=True):
    loss_meter = AvgMeter()
    progress = tqdm(loader, total=len(loader)) if master_process else loader
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        if master_process:
            progress.set_postfix(loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, loader, device, master_process=True):
    loss_meter = AvgMeter()
    progress = tqdm(loader, total=len(loader)) if master_process else loader
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        with torch.no_grad():
            loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        if master_process:
            progress.set_postfix(loss=loss_meter.avg)
    return loss_meter


def save_checkpoint(state: Dict[str, Any], epoch: int, is_best: bool = False) -> None:
    """Save checkpoint to disk"""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(state, checkpoint_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pt')
        torch.save(state, best_path)

def load_latest_checkpoint(device: torch.device) -> tuple[Dict[str, Any], int]:
    """Load the latest checkpoint if it exists"""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    
    if not checkpoint_files:
        return None, 0
        
    # Find the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    epoch = int(latest_checkpoint.split('_')[-1].split('.')[0])
    
    return checkpoint, epoch


def main():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        device = CFG.device
        master_process = True

    if master_process:
        print(f"Using device: {device}")

    image_names, captions = load_flickr_data()
    train_images, train_captions, val_images, val_captions = split_data(image_names, captions)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = build_loaders(train_images, train_captions, val_images, val_captions, tokenizer, ddp=ddp)
    model = CLIPTransferCaptionModelGPT2(gpt_name="gpt2").to(device)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Setup optimizer based on config
    if CFG.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    elif CFG.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    elif CFG.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {CFG.optimizer}")
    
    # Try to load checkpoint
    start_epoch = 0
    best_loss = float("inf")
    if master_process:
        checkpoint, last_epoch = load_latest_checkpoint(device)
        if checkpoint is not None:
            if ddp:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = last_epoch
            best_loss = checkpoint['best_loss']
            print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")

    for epoch in range(start_epoch, CFG.epochs):
        if master_process:
            print(f"\nEpoch {epoch + 1}/{CFG.epochs}")
        if ddp:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, device, master_process)
        
        model.eval()
        valid_loss = valid_epoch(model, val_loader, device, master_process)
        
        if master_process:
            print(f"Train Loss: {train_loss.avg:.4f} | Val Loss: {valid_loss.avg:.4f}")
            is_best = valid_loss.avg < best_loss
            if is_best:
                best_loss = valid_loss.avg
            
            # Save checkpoint every 4 epochs or on the last epoch
            if (epoch + 1) % 4 == 0 or epoch == CFG.epochs - 1:
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if ddp else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'train_loss': train_loss.avg,
                    'valid_loss': valid_loss.avg,
                }
                save_checkpoint(state, epoch + 1, is_best)
                if is_best:
                    print(f"Saved new best model (val_loss: {best_loss:.4f})")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
