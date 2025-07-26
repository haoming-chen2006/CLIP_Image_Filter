import os
import random
import time
from typing import List

import cv2
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from transformers import AutoTokenizer

from modules import TransferHead, GPT

from config import CFG
from dataset import CLIPDataset, get_transforms, load_flickr_data
from clip import CLIPModel
from utils import AvgMeter, get_lr

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


class CLIPTransferCaptionModelGPT2(nn.Module):
    """Generate captions using frozen CLIP image encoder and GPT2-medium from modules."""

    def __init__(self, gpt_name: str = "gpt2-medium"):
        super().__init__()
        # Load CLIP and freeze it
        self.clip = CLIPModel()
        for p in self.clip.parameters():
            p.requires_grad = False

        # Load GPT2-medium from modules with pretrained weights
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
        logits = logits[:, 1:, :]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
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
    for step, batch in enumerate(progress, 1):
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        if master_process:
            progress.set_postfix(loss=loss_meter.avg, lr=get_lr(optimizer))
            print(f"Train step {step}/{len(loader)} - loss: {loss.item():.4f}")
    return loss_meter


def valid_epoch(model, loader, device, master_process=True):
    loss_meter = AvgMeter()
    progress = tqdm(loader, total=len(loader)) if master_process else loader
    for step, batch in enumerate(progress, 1):
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        with torch.no_grad():
            loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        if master_process:
            progress.set_postfix(loss=loss_meter.avg)
            print(f"Valid step {step}/{len(loader)} - loss: {loss.item():.4f}")
    return loss_meter


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

    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = build_loaders(train_images, train_captions, val_images, val_captions, tokenizer, ddp=ddp)

    model = CLIPTransferCaptionModelGPT2().to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=CFG.weight_decay)

    best_loss = float("inf")
    for epoch in range(CFG.epochs):
        if master_process:
            print(f"Epoch {epoch + 1}/{CFG.epochs}")
        if ddp:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, device, master_process)
        model.eval()
        valid_loss = valid_epoch(model, val_loader, device, master_process)
        if master_process:
            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(model.module.state_dict() if ddp else model.state_dict(), "transfer_caption_best.pt")
                print("Saved best model")
            print(f"Train Loss: {train_loss.avg:.4f} | Val Loss: {valid_loss.avg:.4f}")

    # Generation is not implemented for the GPT2 training script

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
