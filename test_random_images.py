import os
import random
from typing import List, Tuple

import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer

from train_transfer_caption_gpt2 import CLIPTransferCaptionModelGPT2
from dataset import get_transforms, load_flickr_data
from eval import generate_caption


def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess an image for the model"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_transforms("valid")
    image = transform(image=image)['image']
    return image.unsqueeze(0)


def evaluate_random_images(num_images: int = 3):
    """Evaluate model on random images from the dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    image_paths, captions = load_flickr_data()
    
    # Randomly sample images
    samples = random.sample(list(enumerate(zip(image_paths, captions))), num_images)
    selected_indices, selected_pairs = zip(*samples)
    selected_images, selected_captions = zip(*selected_pairs)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = CLIPTransferCaptionModelGPT2(gpt_name="gpt2").to(device)
    checkpoint = torch.load("best.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token
    
    print(f"\nGenerating captions for {num_images} random images...")
    print("-" * 50)
    
    # Create figure for visualization
    fig = plt.figure(figsize=(15, 6 * num_images))
    fig.suptitle("Image Captions Comparison", fontsize=16, y=0.95)
    
    results = []
    
    for idx, (image_path, target_caption) in enumerate(zip(selected_images, selected_captions)):
        try:
            print(f"\nProcessing image {idx + 1}/{num_images}")
            print(f"Image path: {image_path}")
            
            # Load and process image
            image_tensor = load_image(image_path)
            
            # Generate caption
            generated_caption = generate_caption(model, tokenizer, image_tensor, device=device)
            results.append((image_path, generated_caption, target_caption))
            
            print(f"✓ Target caption:    \"{target_caption}\"")
            print(f"✓ Generated caption: \"{generated_caption}\"")
            print("-" * 50)
            
            # Create subplot for this image
            plt.subplot(num_images, 1, idx + 1)
            
            # Display image
            img = Image.open(image_path)
            plt.imshow(img)
            
            # Add captions as title
            plt.title(f"Image {idx + 1}\n" + 
                     f"Target: \"{target_caption}\"\n" +
                     f"Generated: \"{generated_caption}\"",
                     fontsize=10, pad=10)
            plt.axis('off')
        
        except Exception as e:
            print(f"✗ Error processing image: {str(e)}")
            print("-" * 50)
    
    # Save visualization
    if results:
        plt.tight_layout()
        plt.savefig('random_captions_comparison.png', bbox_inches='tight', dpi=300)
        print("\nVisualization saved as 'random_captions_comparison.png'")


if __name__ == "__main__":
    # Don't set random seed to get different images each time
    evaluate_random_images(num_images=3)
