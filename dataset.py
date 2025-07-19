import os
import pickle

import albumentations as A
import cv2
import numpy as np
import torch

import config as CFG


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)


class ArtBenchDataset(torch.utils.data.Dataset):
    """Dataset loader for ArtBench10 pickled batches."""

    def __init__(self, root, tokenizer, transforms, train=True):
        self.transforms = transforms
        self.tokenizer = tokenizer

        with open(os.path.join(root, "meta"), "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        self.styles = meta["styles"]

        self.images = []
        self.labels = []

        if train:
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = ["test_batch"]

        for bf in batch_files:
            with open(os.path.join(root, bf), "rb") as f:
                batch = pickle.load(f, encoding="latin1")
            self.images.append(batch["data"])
            self.labels.extend(batch["labels"])

        self.images = np.concatenate(self.images, axis=0).reshape(-1, 3, 32, 32)
        self.captions = [self.styles[idx] for idx in self.labels]

        self.encoded_captions = tokenizer(
            self.captions, padding=True, truncation=True, max_length=CFG.max_length
        )

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encoded_captions.items()
        }
        image = self.images[idx].transpose(1, 2, 0)
        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["caption"] = self.captions[idx]
        return item



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )