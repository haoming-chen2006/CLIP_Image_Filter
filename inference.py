import gc
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import os
from config import CFG
from dataset import CLIPDataset, get_transforms
from clip import CLIPModel


def load_image_list(csv_path: str):
    """Load image filenames from a CSV list"""
    paths = []
    try:
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.append(os.path.basename(line))
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return paths


def load_flickr_data(csv_path: str = "my-app/public/image_paths.csv"):
    """Load image names from a simple CSV file"""
    image_names = load_image_list(csv_path)
    comments = ["" for _ in image_names]
    return image_names, comments


def get_image_embeddings(image_names, comments, model_path):
    """Precompute image embeddings for the dataset"""
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    
    # Create dataset and loader
    dataset = CLIPDataset(
        image_names[:10000],  # Use subset for faster inference
        comments[:10000],
        tokenizer,
        get_transforms("valid"),
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )
    
    # Load model
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing image embeddings"):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    
    return model, torch.cat(valid_image_embeddings), image_names[:10000]


def rank_images(model, image_embeddings, query, image_filenames, n=9):
    """Rank images for a text query and return the top filenames."""
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query], padding=True, truncation=True, max_length=CFG.max_length, return_tensors="pt")

    batch = {key: values.to(CFG.device) for key, values in encoded_query.items()}

    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    _, indices = torch.topk(dot_similarity.squeeze(0), n)
    matches = [image_filenames[idx] for idx in indices]
    return matches


def find_matches(model, image_embeddings, query, image_filenames, n=9):
    """Find images matching a text query and visualize results."""
    matches = rank_images(model, image_embeddings, query, image_filenames, n=n)
    
    # Visualize results
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, match in enumerate(matches):
        if i >= len(axes):
            break
            
        try:
            image_path = f"{CFG.image_path}/{match}"
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                axes[i].imshow(image)
                axes[i].set_title(f"Match {i+1}: {match}", fontsize=10)
            else:
                axes[i].text(0.5, 0.5, f"Image not found:\n{match}", ha='center', va='center')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center')
        
        axes[i].axis("off")
    
    # Hide unused subplots
    for i in range(len(matches), len(axes)):
        axes[i].axis("off")
    
    plt.suptitle(f'Search Results for: "{query}"', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'search_results_{query.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved search results to 'search_results_{query.replace(' ', '_')}.png'")
    
    try:
        plt.show()
    except:
        print("Display not available, saved to file instead")


def main():
    """Main inference function"""
    print("CLIP Inference - Text-to-Image Search")
    print("=" * 50)

    # Load data
    image_names, comments = load_flickr_data()
    CFG.image_path = "my-app/public/images"
    if image_names is None:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(image_names)} images")
    
    # Check if model exists
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Precompute image embeddings
    print("Computing image embeddings...")
    model, image_embeddings, subset_filenames = get_image_embeddings(image_names, comments, model_path)
    
    # Interactive search
    print("\nReady for text-to-image search!")
    print("Enter text queries to find matching images.")
    print("Examples: 'a dog running', 'people on a beach', 'sunset over mountains'")
    print("Type 'quit' to exit.")
    
    while True:
        query = input("\nEnter search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print(f"Searching for: '{query}'")
        find_matches(model, image_embeddings, query, subset_filenames, n=9)


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="CLIP inference")
    parser.add_argument("--query", type=str, help="run a single search query")
    parser.add_argument("--top", type=int, default=1, help="number of results to return")
    parser.add_argument("--encode", type=str, help="encode text and return embedding")
    args = parser.parse_args()

    if args.encode:
        tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
        encoded = tokenizer([args.encode], padding=True, truncation=True, max_length=CFG.max_length, return_tensors="pt")
        model = CLIPModel().to(CFG.device)
        model.load_state_dict(torch.load("best.pt", map_location=CFG.device))
        model.eval()
        batch = {k: v.to(CFG.device) for k, v in encoded.items()}
        with torch.no_grad():
            text_features = model.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            text_embeddings = model.text_projection(text_features)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        print(json.dumps({"embedding": text_embeddings.squeeze(0).cpu().tolist()}))
    elif args.query:
        image_names, comments = load_flickr_data()
        CFG.image_path = "my-app/public/images"
        model_path = "best.pt"
        model, image_embeddings, subset_filenames = get_image_embeddings(image_names, comments, model_path)
        matches = rank_images(model, image_embeddings, args.query, subset_filenames, n=args.top)
        print(json.dumps({"matches": matches}))
    else:
        main()
