import os
import glob
import random
from typing import List, Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from train_transfer_caption_gpt2 import CLIPTransferCaptionModelGPT2
from dataset import get_transforms, load_flickr_data
from config import TransferGPT2Config as CFG
from clip import CLIPModel
from inference import rank_images, get_image_embeddings

# Colors for visualization
COLORS = {
    'target': '#2ecc71',  # Green
    'clip': '#3498db',    # Blue
    'gpt2': '#e74c3c'     # Red
}


def load_latest_checkpoint(device: torch.device) -> dict:
    """Load the latest checkpoint from the checkpoints directory"""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoints found in the checkpoints directory")
        
    # Find the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    return checkpoint


def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess an image for the model"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_transforms("valid")  # Use validation transforms
    image = transform(image=image)['image']
    return image.unsqueeze(0)  # Add batch dimension


def generate_caption(
    model: CLIPTransferCaptionModelGPT2,
    tokenizer: AutoTokenizer,
    image: torch.Tensor,
    max_length: int = 50,
    temperature: float = 1.0,  # Increased temperature for more diverse captions
    top_k: int = 50,  # Add top-k sampling
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> str:
    """Generate a caption for an image using the trained model"""
    model.eval()
    image = image.to(device)
    
    with torch.no_grad():
        # Get image features and convert to text embedding space
        img_feat = model.clip.image_encoder(image)
        clip_embed = model.clip.image_projection(img_feat)
        image_embedding = model.transfer_head(clip_embed)
        
        # Start with the image embedding converted to token embeddings shape
        batch_size = image_embedding.size(0)
        inputs_embeds = image_embedding.view(batch_size, 1, -1)  # [batch_size, 1, hidden_size]
        generated = []
        
        for _ in range(max_length):
            # Get position IDs based on current sequence length
            position_ids = torch.arange(0, inputs_embeds.size(1), dtype=torch.long, device=device)
            pos_embeds = model.lm.transformer.wpe(position_ids).unsqueeze(0).expand(batch_size, -1, -1)
            
            # Combine embeddings and position
            hidden_states = model.lm.transformer.drop(inputs_embeds + pos_embeds)
            
            # Run through transformer blocks
            for block in model.lm.transformer.h:
                block_output = block(hidden_states)
                hidden_states = block_output[0] if isinstance(block_output, tuple) else block_output
            hidden_states = model.lm.transformer.ln_f(hidden_states)
            
            # Get logits for next token prediction
            logits = model.lm.lm_head(hidden_states[:, -1:])
            logits = logits / temperature
            
            # Apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from top-k
            next_token_index = torch.multinomial(probs.squeeze(), num_samples=1)
            next_token = top_k_indices.squeeze()[next_token_index]
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append token to generated sequence
            generated.append(next_token.item())
            
            # Get embeddings for the next iteration
            next_token_embeds = model.lm.transformer.wte(next_token.unsqueeze(0)).view(batch_size, 1, -1)
            inputs_embeds = torch.cat([inputs_embeds, next_token_embeds], dim=1)
    
    # Decode the generated tokens
    caption = tokenizer.decode(generated, skip_special_tokens=True)
    return caption.strip()


def evaluate_on_images(image_paths: List[str], save_visualization: bool = True):
    """Evaluate the model on a list of images and optionally save visualizations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load GPT2 model and tokenizer
    print("Loading GPT2 caption model and tokenizer...")
    gpt2_model = CLIPTransferCaptionModelGPT2(gpt_name="gpt2").to(device)
    checkpoint = load_latest_checkpoint(device)
    gpt2_model.load_state_dict(checkpoint['model_state_dict'])
    gpt2_model.eval()
    
    # Load CLIP model for semantic search
    print("Loading CLIP model...")
    clip_model = CLIPModel().to(device)
    clip_model.load_state_dict(torch.load("best.pt", map_location=device))
    clip_model.eval()
    
    # Load Flickr dataset for getting target captions and CLIP search
    print("Loading Flickr dataset...")
    flickr_images, flickr_captions = load_flickr_data()
    
    # Compute CLIP embeddings for the full dataset
    print("Computing CLIP embeddings for the dataset...")
    _, image_embeddings, subset_filenames = get_image_embeddings(flickr_images, flickr_captions, "best.pt")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token  # Use EOS as BOS token since GPT2 doesn't have explicit BOS
    
    print(f"\nGenerating captions for {len(image_paths)} images...")
    print("-" * 50)
    
    results = []
    
    if save_visualization:
        fig = plt.figure(figsize=(15, 5 * len(image_paths)))
        fig.suptitle("Generated Image Captions", fontsize=16, y=0.95)
    
    for idx, image_path in enumerate(image_paths):
        try:
            print(f"\nProcessing image {idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
            image_name = os.path.basename(image_path)
            
            # 1. Get target caption from Flickr dataset
            try:
                image_idx = flickr_images.index(image_name)
                target_caption = flickr_captions[image_idx]
            except ValueError:
                target_caption = "Target caption not found"
            
            # 2. Get CLIP-based similar caption
            clip_matches = rank_images(clip_model, image_embeddings, "", subset_filenames, n=1)
            if clip_matches:
                try:
                    clip_img_idx = flickr_images.index(clip_matches[0])
                    clip_caption = flickr_captions[clip_img_idx]
                except ValueError:
                    clip_caption = "CLIP caption not found"
            else:
                clip_caption = "No CLIP matches found"
            
            # 3. Generate GPT2 caption
            image_tensor = load_image(image_path)
            gpt2_caption = generate_caption(gpt2_model, tokenizer, image_tensor, device=device)
            
            results.append((image_path, target_caption, clip_caption, gpt2_caption))
            
            print(f"✓ Target caption:  \"{target_caption}\"")
            print(f"✓ CLIP caption:    \"{clip_caption}\"")
            print(f"✓ GPT2 caption:    \"{gpt2_caption}\"")
            print("-" * 50)
            
            if save_visualization:
                # Create subplot for this image
                plt.subplot(len(image_paths), 1, idx + 1)
                
                # Display image
                img = Image.open(image_path)
                plt.imshow(img)
                
                # Add all three captions as title
                plt.title(f"Image: {os.path.basename(image_path)}\n" + 
                         f"Target: \"{target_caption}\"\n" +
