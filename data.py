import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    # Use b'...' for keys to support CIFAR-style encoding
    X = batch['data']
    y = batch['labels']
    filenames = batch.get(b'filenames', [])
    print("Loaded images shape:", X.shape)

    # Reshape: each image is 32x32x3
    X = X.reshape(-1, 3, 256, 256).transpose(0, 2, 3, 1)  # (N, H, W, C)
    return X, y, filenames

# Load the first batch
x, y, filenames = load_batch("/pscratch/sd/h/haoming/Projects/clip/artbench-10-python/artbench-10-batches-py/data_batch_1")

# Plot the first image
plt.imshow(x[0])
plt.title(f"Label: {y[0]}")
plt.axis("off")

# Save the image
plt.savefig("artbench_sample.png", dpi=300, bbox_inches='tight')
plt.show()


