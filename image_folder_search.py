import os
import glob
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A

from config import CFG
from clip import CLIPModel


class ImageFolderDataset(torch.utils.data.Dataset):
    """Dataset for loading images from a folder without captions"""
    
    def __init__(self, image_paths, transforms):
        self.image_paths = image_paths
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            # Create placeholder if image not found
            print(f"Warning: Image {image_path} not found, using placeholder.")
            image = np.zeros((CFG.size, CFG.size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.transforms(image=image)['image']
        
        return {
            'image': torch.tensor(image).permute(2, 0, 1).float(),
            'image_path': image_path
        }


def get_image_transforms():
    """Get image transforms for inference"""
    return A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(max_pixel_value=255.0),
    ])


def load_images_from_folder(folder_path):
    """Load all images from a folder with supported formats"""
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']
    image_paths = []
    
    print(f"Scanning folder: {folder_path}")
    
    for fmt in supported_formats:
        pattern = os.path.join(folder_path, '**', fmt)
        paths = glob.glob(pattern, recursive=True)
        image_paths.extend(paths)
        
        # Also check without recursive for direct files
        pattern = os.path.join(folder_path, fmt)
        paths = glob.glob(pattern)
        image_paths.extend(paths)
    
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    
    print(f"Found {len(image_paths)} images")
    return image_paths


def find_latest_checkpoint():
    """Find the latest checkpoint or best model"""
    # First check for best.pt
    if os.path.exists("best.pt"):
        print("Using best.pt model")
        return "best.pt"
    
    # Look for latest checkpoint
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
        if checkpoints:
            # Sort by modification time and get latest
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Using latest checkpoint: {latest_checkpoint}")
            return latest_checkpoint
    
    return None


def compute_image_embeddings(image_paths, model_path, batch_size=None):
    """Precompute image embeddings for all images in the folder"""
    if batch_size is None:
        batch_size = CFG.batch_size
    
    # Create dataset and loader
    dataset = ImageFolderDataset(image_paths, get_image_transforms())
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )
    
    # Load model
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    image_embeddings = []
    processed_paths = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing image embeddings"):
            # Extract images and paths
            images = batch["image"].to(CFG.device)
            paths = batch["image_path"]
            
            # Compute embeddings
            image_features = model.image_encoder(images)
            batch_embeddings = model.image_projection(image_features)
            
            image_embeddings.append(batch_embeddings.cpu())
            processed_paths.extend(paths)
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(image_embeddings, dim=0)
    
    return model, all_embeddings, processed_paths


def find_matches(model, image_embeddings, image_paths, query, n=9):
    """Find images matching a text query"""
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query], padding=True, truncation=True, max_length=CFG.max_length, return_tensors="pt")
    
    batch = {key: values.to(CFG.device) for key, values in encoded_query.items()}
    
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    # Normalize embeddings
    image_embeddings_n = F.normalize(image_embeddings.to(CFG.device), p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Compute similarity
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    # Get top matches
    scores, indices = torch.topk(dot_similarity.squeeze(0), n)
    matches = [(image_paths[idx], scores[i].item()) for i, idx in enumerate(indices)]
    
    # Visualize results
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    print(f"\nTop {n} matches for '{query}':")
    print("-" * 50)
    
    for i, (image_path, score) in enumerate(matches):
        if i >= len(axes):
            break
        
        print(f"{i+1}. Score: {score:.4f} - {os.path.basename(image_path)}")
        
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axes[i].imshow(image)
                axes[i].set_title(f"#{i+1} (Score: {score:.3f})\n{os.path.basename(image_path)}", fontsize=9)
            else:
                axes[i].text(0.5, 0.5, f"Image not found:\n{os.path.basename(image_path)}", ha='center', va='center')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center')
        
        axes[i].axis("off")
    
    # Hide unused subplots
    for i in range(len(matches), len(axes)):
        axes[i].axis("off")
    
    plt.suptitle(f'Text-to-Image Search Results for: "{query}"', fontsize=16)
    plt.tight_layout()
    
    # Save results
    safe_query = query.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_file = f'search_results_{safe_query}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved search results to '{output_file}'")
    
    try:
        plt.show()
    except:
        print("Display not available, saved to file instead")
    
    return matches


def main():
    """Main function for image folder search"""
    print("CLIP Image Folder Search")
    print("=" * 50)
    
    # Get image folder from user
    while True:
        folder_path = input("Enter the path to the image folder: ").strip()
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            break
        else:
            print(f"Folder '{folder_path}' does not exist or is not a directory. Please try again.")
    
    # Load images from folder
    image_paths = load_images_from_folder(folder_path)
    
    if not image_paths:
        print("No supported images found in the folder!")
        return
    
    # Find model
    model_path = find_latest_checkpoint()
    if model_path is None:
        print("No trained model found!")
        print("Please train the model first using train.py")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Compute image embeddings
    print("Computing image embeddings...")
    try:
        model, image_embeddings, processed_paths = compute_image_embeddings(image_paths, model_path)
        print(f"✓ Computed embeddings for {len(processed_paths)} images")
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        # Try with smaller batch size if CUDA OOM
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory, trying with smaller batch size...")
            try:
                model, image_embeddings, processed_paths = compute_image_embeddings(image_paths, model_path, batch_size=8)
                print(f"✓ Computed embeddings for {len(processed_paths)} images (with reduced batch size)")
            except Exception as e2:
                print(f"Still failed with smaller batch size: {e2}")
                return
        else:
            return
    
    # Interactive search
    print("\n" + "=" * 50)
    print("Ready for text-to-image search!")
    print("Enter text queries to find matching images.")
    print("Examples:")
    print("  - 'a dog running in the park'")
    print("  - 'sunset over mountains'")
    print("  - 'people walking on a beach'")
    print("  - 'a red car'")
    print("  - 'flowers in a garden'")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("=" * 50)
    
    while True:
        query = input("\nEnter search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print(f"\nSearching for: '{query}'")
        try:
            matches = find_matches(model, image_embeddings, processed_paths, query, n=9)
        except Exception as e:
            print(f"Error during search: {e}")
            continue


if __name__ == "__main__":
    main()
