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

from transformers import AutoTokenizer, AutoModelForCausalLM

from config import CFG
from dataset import CLIPDataset, get_transforms, load_flickr_data
from clip import CLIPModel
from modules import TransferHead
from utils import AvgMeter, get_lr


class CLIPTransferCaptionModel(nn.Module):
    """Generate captions using frozen CLIP image encoder and GPT2-medium."""

    def __init__(self, gpt_name: str = "gpt2-medium"):
        super().__init__()
        # Load CLIP and freeze it
        self.clip = CLIPModel()
        for p in self.clip.parameters():
            p.requires_grad = False

        # Load GPT2-medium
        self.lm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

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
        inputs_embeds = torch.cat([prefix, token_embeds[:, :-1, :]], dim=1)
        attn_mask = torch.cat([
            torch.ones(prefix.size(0), 1, device=attention_mask.device),
            attention_mask[:, :-1],
        ], dim=1)

        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=input_ids)
        return outputs.loss

    @torch.no_grad()
    def generate_captions(self, image_paths: List[str], tokenizer, max_length: int = 50):
        """Generate captions for a list of image paths."""
        self.eval()
        images = []
        for p in image_paths:
            img = cv2.imread(os.path.join(CFG.image_path, p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = get_transforms("valid")(image=img)["image"]
            images.append(torch.tensor(img).permute(2, 0, 1).float())

        if not images:
            return []
        batch = torch.stack(images).to(next(self.parameters()).device)
        img_feat = self.clip.image_encoder(batch)
        clip_embed = self.clip.image_projection(img_feat)
        prefix = self.transfer_head(clip_embed)
        attention_mask = torch.ones(prefix.size(0), 1, device=prefix.device)
        generated = self.lm.generate(inputs_embeds=prefix.unsqueeze(1), attention_mask=attention_mask, max_length=max_length)
        captions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        self.train()
        return captions


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

    model = CLIPTransferCaptionModel().to(device)
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

    raw_model = model.module if ddp else model
    sample_images = val_images[:5]
    captions_out = raw_model.generate_captions(sample_images, tokenizer)
    if master_process:
        for img, cap in zip(sample_images, captions_out):
            print(f"{img} -> {cap}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
