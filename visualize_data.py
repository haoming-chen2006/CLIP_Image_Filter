import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import config as CFG


def load_sample(index=0):
    root = CFG.dataset_root
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


def main():
    image, caption = load_sample(0)
    plt.imshow(image)
    plt.title(caption)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
