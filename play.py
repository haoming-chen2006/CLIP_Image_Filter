from inference import get_embed
import torch

image1 = "/pscratch/sd/h/haoming/Projects/clip/comp1.jpg"
image2 = "/pscratch/sd/h/haoming/Projects/clip/comp2.jpg"

print("Loading embeddings for both images...")

# Get embeddings for both images
embed1 = get_embed(image1)
embed2 = get_embed(image2)

print(f"Image 1 embedding shape: {embed1.shape}")
print(f"Image 2 embedding shape: {embed2.shape}")

queryies = ["headphones","hat","sunshine"]
tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query], padding=True, truncation=True, max_length=CFG.max_length, return_tensors="pt")
similarity = torch.cosine_similarity(embed1, embed2, dim=0)
print(f"Cosine similarity between the two images: {similarity.item():.4f}")

# Print some statistics
print(f"Image 1 embedding norm: {torch.norm(embed1).item():.4f}")
print(f"Image 2 embedding norm: {torch.norm(embed2).item():.4f}")
print(f"Image 1 embedding mean: {embed1.mean().item():.4f}")
print(f"Image 2 embedding mean: {embed2.mean().item():.4f}")