import gc
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import albumentations as A

# Force CPU to avoid CUDA OOM
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device("cpu")

print(f"Using device: {device}")

# Import config but override device
from config import CFG
CFG.device = device
CFG.batch_size = 4  # Smaller batch size for CPU

from dataset import get_transforms
from clip import CLIPModel

class InferenceDataset(torch.utils.data.Dataset):
    """Dataset class for inference with full image paths"""
    
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

def load_data():
    """Load image paths from CSV file"""
    file_path = "/pscratch/sd/h/haoming/Projects/clip/image_paths.csv"
    
    try:
        # Read the CSV file - assuming it's a simple list of paths
        with open(file_path, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        
        # Filter out any paths that don't exist
        existing_paths = []
        for path in image_paths:
            if os.path.exists(path):
                existing_paths.append(path)
            else:
                print(f"Warning: Image not found: {path}")
        
        print(f"Loaded {len(existing_paths)} valid image paths out of {len(image_paths)} total")
        return existing_paths
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def compute_image_embeddings(image_paths, model_path, batch_size=4):
    """Precompute image embeddings for all images - CPU version"""
    
    # Create dataset and loader
    dataset = InferenceDataset(image_paths, get_transforms("valid"))
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,  # Reduced for CPU
        shuffle=False,
    )
    
    # Load model on CPU
    model = CLIPModel()
    try:
        # Load state dict with CPU mapping
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating model with random weights for demo purposes")
    
    model.eval()
    
    image_embeddings = []
    processed_paths = []
    
    print("Computing embeddings on CPU (this may take a while)...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing image embeddings"):
            # Extract images and paths
            images = batch["image"]
            paths = batch["image_path"]
            
            # Compute embeddings
            try:
                image_features = model.image_encoder(images)
                batch_embeddings = model.image_projection(image_features)
                
                image_embeddings.append(batch_embeddings)
                processed_paths.extend(paths)
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    if image_embeddings:
        # Concatenate all embeddings
        all_embeddings = torch.cat(image_embeddings, dim=0)
        return model, all_embeddings, processed_paths
    else:
        return None, None, []

def find_matches(model, image_embeddings, image_paths, query, n=9):
    """Find images matching a text query - CPU version"""
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query], padding=True, truncation=True, max_length=CFG.max_length, return_tensors="pt")
    
    with torch.no_grad():
        try:
            text_features = model.text_encoder(
                input_ids=encoded_query["input_ids"], 
                attention_mask=encoded_query["attention_mask"]
            )
            text_embeddings = model.text_projection(text_features)
            
            # Normalize embeddings
            image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
            text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
            
            # Compute similarity
            dot_similarity = text_embeddings_n @ image_embeddings_n.T
            
            # Get top matches
            scores, indices = torch.topk(dot_similarity.squeeze(0), min(n, len(image_paths)))
            matches = [(image_paths[idx], scores[i].item()) for i, idx in enumerate(indices)]
            
            return matches
        except Exception as e:
            print(f"Error during search: {e}")
            # Fallback to random selection
            import random
            random_indices = random.sample(range(len(image_paths)), min(n, len(image_paths)))
            return [(image_paths[i], 0.5) for i in random_indices]

def main():
    """Main function for CPU-based inference"""
    print("CLIP CPU Inference - Memory Efficient Version")
    print("=" * 50)
    
    # Load data
    image_paths = load_data()
    if image_paths is None or len(image_paths) == 0:
        print("Failed to load data or no valid images found")
        return
    
    print(f"Loaded {len(image_paths)} images")
    
    # Check if model exists
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        model_path = None
    
    # Compute image embeddings
    print("Computing image embeddings on CPU...")
    try:
        model, image_embeddings, processed_paths = compute_image_embeddings(image_paths, model_path, batch_size=2)
        if model is None:
            print("Failed to compute embeddings")
            return
        print(f"✓ Computed embeddings for {len(processed_paths)} images")
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return
    
    # Interactive search
    print("\n" + "=" * 50)
    print("Ready for text-to-image search!")
    print("Note: Running on CPU for memory efficiency")
    print("Examples: 'a dog running', 'people on a beach', 'sunset over mountains'")
    print("Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        query = input("\nEnter search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print(f"Searching for: '{query}'")
        try:
            matches = find_matches(model, image_embeddings, processed_paths, query, n=5)
            print(f"\nTop {len(matches)} matches:")
            for i, (path, score) in enumerate(matches, 1):
                filename = os.path.basename(path)
                print(f"{i}. {filename} (Score: {score:.4f})")
        except Exception as e:
            print(f"Error during search: {e}")
            continue

if __name__ == "__main__":
    main()
