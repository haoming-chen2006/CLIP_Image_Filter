import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from config import CFG


def load_sample(index=0, train=True):
    """Load a sample from ArtBench10.

    Supports both the binary (32x32) and the image-folder (256x256) versions
    of the dataset depending on ``CFG.dataset_type``.
    """
    root = CFG.dataset_root

    if CFG.dataset_type == "binary":
        with open(os.path.join(root, "meta"), "rb") as f:
            meta = pickle.load(f, encoding="latin1")
        styles = meta["styles"]

        with open(os.path.join(root, "data_batch_1"), "rb") as f:
            batch = pickle.load(f, encoding="latin1")

        images = batch["data"].reshape(-1, 3, 32, 32)
        labels = batch["labels"]

        img = images[index].transpose(1, 2, 0)
        caption = styles[labels[index]]
        return img.astype(np.uint8), caption

    # Image folder variant (e.g. 256x256)
    split = "train" if train else "test"
    split_root = os.path.join(root, split)
    style_dirs = [d for d in sorted(os.listdir(split_root)) if os.path.isdir(os.path.join(split_root, d))]

    image_paths = []
    captions = []
    for style in style_dirs:
        sdir = os.path.join(split_root, style)
        for fname in os.listdir(sdir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(sdir, fname))
                captions.append(style)

    img_path = image_paths[index]
    caption = captions[index]
    img = plt.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img.astype(np.uint8), caption


def main():
    image, caption = load_sample(0)
    plt.imshow(image)
    plt.title(caption)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
