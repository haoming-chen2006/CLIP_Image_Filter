import kagglehub
import shutil
import os

# Download to default kaggle location first
print("Downloading dataset...")
path = kagglehub.dataset_download("ikarus777/best-artworks-of-all-time")

print("Downloaded to:", path)

# Move to desired location
custom_path = "/pscratch/sd/h/haoming/Projects/clip/artworks"
if os.path.exists(custom_path):
    print(f"Removing existing directory: {custom_path}")
    shutil.rmtree(custom_path)

print(f"Moving dataset to: {custom_path}")
shutil.move(path, custom_path)

print("✓ Dataset ready at:", custom_path)
