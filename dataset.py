import os
import pandas as pd
import albumentations as A
import cv2
import numpy as np
import torch
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
from PIL import Image

from config import CFG




class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        Simple CLIP dataset
        image_filenames and captions must have the same length
        """
        # Validate and filter image paths
        valid_indices = []
        missing_images = []
        for idx, img_name in enumerate(image_filenames):
            img_path = os.path.join(CFG.image_path, img_name)
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                missing_images.append(img_name)
        
        if missing_images:
            print(f"\nWARN: Found {len(missing_images)} missing images out of {len(image_filenames)}")
            print(f"First few missing images: {missing_images[:5]}")
            print(f"Using {len(valid_indices)} valid images for training/validation")
        
        # Keep only valid images and their captions
        self.image_filenames = [image_filenames[i] for i in valid_indices]
        captions = [captions[i] for i in valid_indices]
        
        self.captions = list(captions)
        
        # Clean captions - ensure all are strings and not None/NaN
        clean_captions = []
        for caption in self.captions:
            if caption is None or str(caption).lower() == 'nan':
                clean_captions.append("No caption available")
            else:
                clean_captions.append(str(caption))
        
        self.captions = clean_captions
        
        if tokenizer is not None:
            self.encoded_captions = tokenizer(
                self.captions, padding=True, truncation=True, max_length=CFG.max_length
            )
        else:
            self.encoded_captions = self.captions        
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        
        image_path = os.path.join(CFG.image_path, self.image_filenames[idx])
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transforms(image=image)
        image = transformed['image']
        
        # Debug checks
        if idx == 0:  # Only print for first item to avoid spam
            print(f"\nImage debug info:")
            print(f"Raw image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Image value range: [{image.min():.3f}, {image.max():.3f}]")
        
        item['image'] = image
        item['caption'] = self.captions[idx]
        
        return item

    def __len__(self):
        return len(self.captions)
    
    def get_stats(self):
        """Return dataset loading statistics"""
        images_loaded = self.total_accessed - self.placeholder_count
        return {
            'total_accessed': self.total_accessed,
            'images_loaded': images_loaded,
            'placeholders_used': self.placeholder_count,
            'success_rate': (images_loaded / self.total_accessed * 100) if self.total_accessed > 0 else 0
        }


def print_dataset_stats(dataset):
    """Print dataset loading statistics"""
    stats = dataset.get_stats()
    print(f"Dataset Loading Statistics:")
    print(f"  Images loaded successfully: {stats['images_loaded']}")
    print(f"  Placeholders used: {stats['placeholders_used']}")
    print(f"  Total accessed: {stats['total_accessed']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")


def load_flickr_data():
    """
    Load Flickr data from results.csv
    Returns: image_names (list), captions (list)
    """
    csv_path = "/pscratch/sd/h/haoming/Projects/clip/flickr/results.csv"
    
    # Load CSV with pipe separator
    df = pd.read_csv(csv_path, sep='|')
    
    # Extract columns and remove any rows with missing data
    df = df.dropna(subset=['image_name', ' comment'])
    
    image_names = df['image_name'].tolist()
    captions = df[' comment'].tolist()  # Note: space before 'comment'
    
    # Additional cleaning - ensure all captions are strings
    clean_image_names = []
    clean_captions = []
    
    for img, cap in zip(image_names, captions):
        if img and cap and str(cap).strip() and str(cap).lower() != 'nan':
            clean_image_names.append(str(img))
            clean_captions.append(str(cap).strip())
    
    print(f"Loaded {len(clean_image_names)} valid image-caption pairs")
    return clean_image_names, clean_captions


def load_instagram_data(
    csv_path="/pscratch/sd/h/haoming/Projects/clip/artworks/instagram_data/captions.csv",
    image_col="image_name",
    caption_col="caption",
):
    """Load Instagram artwork data from a CSV file.

    The CSV file is expected to be comma separated and contain at least two
    columns: one with the image filename and another with the corresponding
    caption.  Rows without a valid caption are dropped.  The function returns
    the list of image filenames and their captions.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if image_col not in df.columns or caption_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{image_col}' and '{caption_col}'"
        )

    df = df.dropna(subset=[image_col, caption_col])
    df = df[df[caption_col].astype(str).str.strip() != ""]

    image_names = df[image_col].astype(str).tolist()
    captions = df[caption_col].astype(str).tolist()

    print(f"Loaded {len(image_names)} valid image-caption pairs from Instagram data")
    return image_names, captions


def get_transforms(mode="train"):
    """Get image transforms"""
    return A.Compose([
        A.Resize(224, 224),  # ResNet expects 224x224 images
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],    # ImageNet stds
            max_pixel_value=255.0
        ),
        A.pytorch.ToTensorV2(),  # Convert to tensor and transpose channels
    ])

def visualize_first_five():
    """Visualize first 5 images and captions from the dataset"""
    
    # Load data
    images, captions = load_flickr_data()
    
    # Create tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Show first 5 filenames and captions
    print("First 5 image filenames and captions:")
    print("=" * 60)
    for i in range(5):
        print(f"{i+1}. Image: {images[i]}")
        print(f"   Caption: {captions[i]}")
        print()
    
    # Create visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i in range(5):
        # Build image path
        img_path = os.path.join(CFG.image_path, images[i])
        
        try:
            # Load image
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[i].imshow(img)
                
                # Truncate caption for title
                caption = captions[i]
                if len(caption) > 40:
                    caption = caption[:37] + "..."
                axes[i].set_title(caption, fontsize=8, wrap=True)
                print(f"✓ Loaded: {images[i]}")
            else:
                axes[i].text(0.5, 0.5, f"Not found:\n{images[i]}", ha='center', va='center')
                axes[i].set_title("Image not found", fontsize=8)
                print(f"✗ Not found: {img_path}")
                
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center')
            axes[i].set_title("Error", fontsize=8)
            print(f"✗ Error loading {images[i]}: {e}")
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("First 5 Images from Flickr Dataset", fontsize=14, y=1.02)
    
    # Save the plot
    plt.savefig('first_five_images.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'first_five_images.png'")
    
    # Try to show (might not work in server environment)
    try:
        plt.show()
    except:
        print("Display not available, image saved to file instead")


# Update the test code at the bottom
if __name__ == "__main__":
    # Test the visualization
    visualize_first_five()
