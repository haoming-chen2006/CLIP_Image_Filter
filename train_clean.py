import os
import torch
from transformers import DistilBertTokenizer

from config import CFG
from dataset import ArtBenchDataset, ArtBenchImageFolderDataset, get_transforms
from CLIP import CLIPModel


def main():
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    dataset_cls = ArtBenchImageFolderDataset if CFG.dataset_type == "folder" else ArtBenchDataset
    train_dataset = dataset_cls(CFG.dataset_root, tokenizer, get_transforms("train"), train=True)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
    )

    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": list(model.image_projection.parameters()) + list(model.text_projection.parameters()), "lr": CFG.head_lr},
        ],
        weight_decay=CFG.weight_decay,
    )

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(CFG.epochs):
        model.train()
        for batch in loader:
            batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            f"checkpoints/clean_epoch_{epoch + 1}.pt",
        )
        print(f"Epoch {epoch + 1} completed. Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
