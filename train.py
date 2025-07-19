import os
import random
from tqdm import tqdm

import torch
from transformers import DistilBertTokenizer

from config import CFG
from dataset import CLIPDataset, get_transforms, load_flickr_data
from clip import CLIPModel
from utils import AvgMeter, get_lr


def split_data(image_names, captions, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test"""
    data = list(zip(image_names, captions))
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    
    train_images, train_captions = zip(*train_data)
    val_images, val_captions = zip(*val_data)
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    return list(train_images), list(train_captions), list(val_images), list(val_captions)


def build_loaders(train_images, train_captions, val_images, val_captions, tokenizer):
    """Build data loaders - exactly following dataset.py approach"""
    
    train_dataset = CLIPDataset(train_images, train_captions, tokenizer, get_transforms("train"))
    val_dataset = CLIPDataset(val_images, val_captions, tokenizer, get_transforms("valid"))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=False
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    
    return loss_meter


def main():
    print("CLIP Training")
    print("=" * 50)
    
    # Load data using the exact same function as dataset.py
    image_names, captions = load_flickr_data()
    
    # Split data
    train_images, train_captions, val_images, val_captions = split_data(image_names, captions)
    
    # Create tokenizer - exactly like dataset.py
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    
    # Build loaders
    train_loader, val_loader = build_loaders(train_images, train_captions, val_images, val_captions, tokenizer)

    # Model and optimizer
    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": list(model.image_projection.parameters()) + list(model.text_projection.parameters()), 
         "lr": CFG.head_lr},
    ], weight_decay=CFG.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    # Training loop
    best_loss = float("inf")
    for epoch in range(CFG.epochs):
        print(f"\nEpoch: {epoch + 1}/{CFG.epochs}")
        
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, "epoch")
        
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, val_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("✓ Saved Best Model!")
        
        print(f"Train Loss: {train_loss.avg:.4f} | Val Loss: {valid_loss.avg:.4f}")

    print("\n✓ Training completed!")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()