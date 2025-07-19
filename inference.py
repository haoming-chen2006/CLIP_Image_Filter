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


def load_flickr_data():
    """Load Flickr data for inference"""
    file_path = "/pscratch/sd/h/haoming/Projects/clip/flickr-dataset/flickr30k_images/results.csv"
    
    try:
        df = pd.read_csv(file_path, sep='|')
        image_names = df['image_name'].dropna().tolist()
        comments = df[' comment'].dropna().tolist()
        return image_names, comments
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


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


def find_matches(model, image_embeddings, query, image_filenames, n=9):
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
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Compute similarity
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    # Get top matches
    _, indices = torch.topk(dot_similarity.squeeze(0), n)
    matches = [image_filenames[idx] for idx in indices]
    
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
    main()