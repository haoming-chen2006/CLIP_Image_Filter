import os
import random
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from config import CFG
from dataset import CLIPDataset, get_transforms, load_flickr_data
from modules import ImageEncoder, GPT2
from utils import AvgMeter, get_lr


class ImageCaptionModelGPT2(nn.Module):
    """Trainable GPT2 caption model with image prefix."""

    def __init__(self, vocab_size, n_embed=768, n_blocks=12, num_heads=12, block_size=128, hidden_dim=3072):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=True, trainable=False)
        self.mlp = nn.Linear(CFG.image_embedding, n_embed)
        self.lm = GPT2(n_blocks, vocab_size, n_embed, block_size, num_heads, hidden_dim)

    def forward(self, batch):
        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        img_feat = self.image_encoder(images)
        prefix = self.mlp(img_feat).unsqueeze(1)

        token_embeds = self.lm.transformer.wte(input_ids)
        embeds = torch.cat([prefix, token_embeds[:, :-1, :]], dim=1)
        mask = torch.cat([
            torch.ones(prefix.size(0), 1, device=attention_mask.device),
            attention_mask[:, :-1],
        ], dim=1)

        logits = self.lm(inputs_embeds=embeds, attention_mask=mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1), ignore_index=0)
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


def build_loaders(train_images, train_captions, val_images, val_captions, tokenizer):
    train_dataset = CLIPDataset(train_images, train_captions, tokenizer, get_transforms("train"))
    val_dataset = CLIPDataset(val_images, val_captions, tokenizer, get_transforms("valid"))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=False
    )
    return train_loader, val_loader


def train_epoch(model, loader, optimizer):
    loss_meter = AvgMeter()
    progress = tqdm(loader, total=len(loader))
    for batch in progress:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        progress.set_postfix(loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, loader):
    loss_meter = AvgMeter()
    progress = tqdm(loader, total=len(loader))
    for batch in progress:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        with torch.no_grad():
            loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        progress.set_postfix(loss=loss_meter.avg)
    return loss_meter


def main():
    image_names, captions = load_flickr_data()
    train_images, train_captions, val_images, val_captions = split_data(image_names, captions)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = build_loaders(train_images, train_captions, val_images, val_captions, tokenizer)

    model = ImageCaptionModelGPT2(vocab_size=len(tokenizer)).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=CFG.weight_decay)

    best_loss = float("inf")
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}/{CFG.epochs}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer)
        model.eval()
        valid_loss = valid_epoch(model, val_loader)
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "gpt2_full_best.pt")
            print("Saved best model")
        print(f"Train Loss: {train_loss.avg:.4f} | Val Loss: {valid_loss.avg:.4f}")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
