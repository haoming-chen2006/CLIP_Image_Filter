import kagglehub
import os

# Set custom cache directory to current working directory
custom_cache = os.path.join(os.getcwd(), "kaggle_cache")
os.environ['KAGGLE_DATA_PROXY_TOKEN'] = ''  # Clear any proxy settings
os.environ['KAGGLEHUB_CACHE_DIR'] = custom_cache

# Download dataset to custom cache location
print("Downloading Flickr dataset...")
path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
print(f"Dataset downloaded to: {path}")

# Create symlink in current directory for easy access
symlink_path = os.path.join(os.getcwd(), "flickr-dataset")
if os.path.exists(symlink_path):
    os.remove(symlink_path)

try:
    os.symlink(path, symlink_path)
    print(f"Created symlink: {symlink_path} -> {path}")
except OSError:
    # If symlink fails, just print the path
    print(f"Use this path to access the dataset: {path}")

# List dataset contents
print("\nDataset structure:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files only
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files) - 5} more files")
