import os
import random
from tqdm import tqdm

import torch
from torch import nn
from transformers import DistilBertTokenizer
from PIL import Image
import matplotlib.pyplot as plt

import config as CFG
from dataset import CLIPDataset, get_transforms, load_flickr8k
from CLIP import CLIPModel
from utils import AvgMeter, get_lr


def show_samples(image_root, image_files, captions, num=6):
    """Visualize a few image-caption pairs."""
    idxs = random.sample(range(len(captions)), min(num, len(captions)))
    cols = 3
    rows = (num + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    for ax, i in zip(axes, idxs):
        path = image_files[i]
        if not os.path.isabs(path):
            path = os.path.join(image_root, path)
        image = Image.open(path)
        ax.imshow(image)
        ax.set_title(captions[i], fontsize=8)
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def build_loaders(tokenizer):
    image_root, train_images, train_captions = load_flickr8k(CFG.captions_path, "train")
    _, val_images, val_captions = load_flickr8k(CFG.captions_path, "val")

    train_dataset = CLIPDataset(
        train_images,
        train_captions,
        tokenizer,
        get_transforms("train"),
        image_root=image_root,
    )
    valid_dataset = CLIPDataset(
        val_images,
        val_captions,
        tokenizer,
        get_transforms("valid"),
        image_root=image_root,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )

    return train_loader, valid_loader


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
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader, valid_loader = build_loaders(tokenizer)

    # visualize a few training samples
    img_root, sample_imgs, sample_caps = load_flickr8k(CFG.captions_path, "train")
    show_samples(img_root, sample_imgs, sample_caps)


    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": list(model.image_projection.parameters()) + list(model.text_projection.parameters()), "lr": CFG.head_lr},
        ],
        weight_decay=CFG.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float("inf")
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            f"checkpoints/epoch_{epoch + 1}.pt",
        )

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")


if __name__ == "__main__":
    main()