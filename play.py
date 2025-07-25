from inference import get_embed, load_single_image_embedding
import torch
import os
from transformers import DistilBertTokenizer
from config import CFG
from clip import CLIPModel
import torch.nn.functional as F

# Use existing images
image1 = "/pscratch/sd/h/haoming/Projects/clip/comp1.jpg"
image2 = "/pscratch/sd/h/haoming/Projects/clip/comp2.jpg"

print("Checking if image files exist...")
print(f"Image 1 exists: {os.path.exists(image1)}")
print(f"Image 2 exists: {os.path.exists(image2)}")

if not os.path.exists(image1) or not os.path.exists(image2):
    print("Available image files in current directory:")
    for f in os.listdir("/pscratch/sd/h/haoming/Projects/clip/"):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"  {f}")
    exit(1)

print("Loading embeddings for both images...")

flickr_image = "/pscratch/sd/h/haoming/Projects/clip/flickr/flickr30k_images/flickr30k_images/1000092795.jpg"

def load_or_fallback(path, fallback=None):
    try:
        return load_single_image_embedding(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        if fallback is not None:
            print(f"Using fallback image: {fallback}")
            return load_single_image_embedding(fallback)
        raise

embed1 = load_or_fallback(image1, flickr_image)
embed2 = load_or_fallback(image2, flickr_image)
embed3 = embed1 - embed2  # Difference in latent space

print(f"Image 1 embedding shape: {embed1.shape}")
print(f"Image 2 embedding shape: {embed2.shape}")
print(f"Embed 3 (difference) shape: {embed3.shape}")

# Compute embedding statistics
print(f"Embedding difference norm (embed3): {torch.norm(embed3).item():.4f}")

# Image similarities
similarity12 = torch.cosine_similarity(embed1, embed2, dim=0)
print(f"Cosine similarity between image 1 and 2: {similarity12.item():.4f}")

# Note: embed3 is a difference vector, so cosine similarity with original embeddings 
# represents how much the difference aligns with each original embedding
similarity_diff_1 = torch.cosine_similarity(embed3, embed1, dim=0)
similarity_diff_2 = torch.cosine_similarity(embed3, embed2, dim=0)
print(f"Cosine similarity between difference and image 1: {similarity_diff_1.item():.4f}")
print(f"Cosine similarity between difference and image 2: {similarity_diff_2.item():.4f}")

# Query similarity
queries = ["headphones", "hat", "sunshine","a boy with a hat","a girl on a beach","mountains and lions"]
print("\nComputing text-image similarities...")

# Load model for text encoding
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")
model = CLIPModel().to(CFG.device)
model.load_state_dict(torch.load(model_path, map_location=CFG.device))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

for query in queries:
    print(f"\nQuery: '{query}'")
    
    # Encode text
    encoded_query = tokenizer([query], padding=True, truncation=True, max_length=CFG.max_length, return_tensors="pt")
    batch = {key: values.to(CFG.device) for key, values in encoded_query.items()}
    
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        text_embedding = model.text_projection(text_features)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1).squeeze(0)
    
    # Calculate similarities with images and difference vector
    sim1 = torch.cosine_similarity(embed1, text_embedding, dim=0)
    sim2 = torch.cosine_similarity(embed2, text_embedding, dim=0)
    sim3_diff = torch.cosine_similarity(embed3, text_embedding, dim=0)  # Similarity with difference vector
    
    print(f"  Similarity with image 1: {sim1.item():.4f}")
    print(f"  Similarity with image 2: {sim2.item():.4f}")
    print(f"  Similarity with difference (embed3): {sim3_diff.item():.4f}")
    
    # Find best match among actual images
    similarities = [sim1.item(), sim2.item()]
    best_idx = similarities.index(max(similarities))
    print(f"  Best image match: Image {best_idx + 1} (similarity: {similarities[best_idx]:.4f})")
    
    # Analyze what the difference vector represents
    if abs(sim3_diff.item()) > 0.1:  # Threshold for meaningful similarity
        direction = "positive" if sim3_diff.item() > 0 else "negative"
        print(f"  The difference vector has {direction} alignment with '{query}' (similarity: {sim3_diff.item():.4f})")

# Print some statistics
print(f"\nEmbedding Statistics:")
print(f"Image 1 embedding norm: {torch.norm(embed1).item():.4f}")
print(f"Image 2 embedding norm: {torch.norm(embed2).item():.4f}")
print(f"Difference (embed3) norm: {torch.norm(embed3).item():.4f}")
print(f"Image 1 embedding mean: {embed1.mean().item():.4f}")
print(f"Image 2 embedding mean: {embed2.mean().item():.4f}")
print(f"Difference (embed3) mean: {embed3.mean().item():.4f}")