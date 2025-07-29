import os
import glob
from typing import List

import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from train_transfer_caption import CLIPTransferCaptionModel
from train_transfer_caption_gpt2 import (
    CLIPTransferCaptionModelGPT2,
    load_latest_checkpoint as load_gpt2_checkpoint,
)
from dataset import get_transforms, load_flickr_data
from clip import CLIPModel
from inference import rank_images, get_image_embeddings

COLORS = {
    'target': '#2ecc71',  # Green
    'gpt2': '#e74c3c',    # Red
    'tiny': '#9b59b6'     # Purple
}

def load_tiny_checkpoint(device: torch.device) -> dict:
    """Load TinyLlama checkpoint saved during training"""
    ckpt_path = os.path.join(os.path.dirname(__file__), 'transfer_caption_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"TinyLlama checkpoint not found at {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)

def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess an image for the models"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_transforms('valid')
    image = transform(image=image)['image']
    return image.unsqueeze(0)

def generate_caption_gpt2(
    model: CLIPTransferCaptionModelGPT2,
    tokenizer: AutoTokenizer,
    image: torch.Tensor,
    device: torch.device,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Generate caption using the GPT2-based model"""
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        img_feat = model.clip.image_encoder(image)
        clip_embed = model.clip.image_projection(img_feat)
        image_embedding = model.transfer_head(clip_embed)
        batch_size = image_embedding.size(0)
        inputs_embeds = image_embedding.view(batch_size, 1, -1)
        generated = []
        for _ in range(max_length):
            position_ids = torch.arange(0, inputs_embeds.size(1), dtype=torch.long, device=device)
            pos_embeds = model.lm.transformer.wpe(position_ids).unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states = model.lm.transformer.drop(inputs_embeds + pos_embeds)
            for block in model.lm.transformer.h:
                block_output = block(hidden_states)
                hidden_states = block_output[0] if isinstance(block_output, tuple) else block_output
            hidden_states = model.lm.transformer.ln_f(hidden_states)
            logits = model.lm.lm_head(hidden_states[:, -1:])
            logits = logits / temperature
            top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_index = torch.multinomial(probs.squeeze(), num_samples=1)
            next_token = top_k_indices.squeeze()[next_token_index]
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated.append(next_token.item())
            next_token_embeds = model.lm.transformer.wte(next_token.unsqueeze(0)).view(batch_size, 1, -1)
            inputs_embeds = torch.cat([inputs_embeds, next_token_embeds], dim=1)
    caption = tokenizer.decode(generated, skip_special_tokens=True)
    return caption.strip()

def generate_caption_tiny(
    model: CLIPTransferCaptionModel,
    tokenizer: AutoTokenizer,
    image: torch.Tensor,
    device: torch.device,
    max_length: int = 50,
) -> str:
    """Generate caption using the TinyLlama-based model"""
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        img_feat = model.clip.image_encoder(image)
        clip_embed = model.clip.image_projection(img_feat)
        prefix = model.transfer_head(clip_embed)
        attention_mask = torch.ones(prefix.size(0), 1, device=device)
        generated = model.lm.generate(
            inputs_embeds=prefix.unsqueeze(1),
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    caption = tokenizer.decode(generated[0], skip_special_tokens=True)
    return caption.strip()

def evaluate_on_images(image_paths: List[str], save_visualization: bool = True):
    """Compare TinyLlama and GPT2 caption models on a list of images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading TinyLlama caption model...")
    tiny_model = CLIPTransferCaptionModel().to(device)
    checkpoint = load_tiny_checkpoint(device)
    print("TinyLlama checkpoint loaded")
    tiny_model.load_state_dict(checkpoint)
    tiny_model.eval()

    print("Loading GPT2 caption model...")
    gpt2_model = CLIPTransferCaptionModelGPT2(gpt_name='gpt2').to(device)
    gpt2_ckpt_tuple = load_gpt2_checkpoint(device)
    if isinstance(gpt2_ckpt_tuple, tuple):
        gpt2_ckpt, _ = gpt2_ckpt_tuple
    else:
        gpt2_ckpt = gpt2_ckpt_tuple
    gpt2_model.load_state_dict(gpt2_ckpt['model_state_dict'])
    gpt2_model.eval()

    print("Loading CLIP model for ranking...")
    clip_model = CLIPModel().to(device)
    clip_model.load_state_dict(torch.load('best.pt', map_location=device))
    clip_model.eval()

    print("Loading Flickr dataset...")
    flickr_images, flickr_captions = load_flickr_data()

    print("Computing CLIP embeddings for the dataset...")
    _, image_embeddings, subset_filenames = get_image_embeddings(flickr_images, flickr_captions, 'best.pt')

    llama_tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_tokenizer.bos_token = gpt2_tokenizer.eos_token

    results = []
    if save_visualization:
        fig = plt.figure(figsize=(15, 5 * len(image_paths)))
        fig.suptitle('TinyLlama vs GPT2 Captions', fontsize=16, y=0.95)

    for idx, image_path in enumerate(image_paths):
        try:
            print(f"\nProcessing image {idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}")
            image_name = os.path.basename(image_path)
            try:
                image_idx = flickr_images.index(image_name)
                target_caption = flickr_captions[image_idx]
            except ValueError:
                target_caption = 'Target caption not found'

            clip_matches = rank_images(clip_model, image_embeddings, '', subset_filenames, n=1)
            if clip_matches:
                try:
                    clip_img_idx = flickr_images.index(clip_matches[0])
                    clip_caption = flickr_captions[clip_img_idx]
                except ValueError:
                    clip_caption = 'CLIP caption not found'
            else:
                clip_caption = 'No CLIP matches found'

            image_tensor = load_image(image_path)
            gpt2_caption = generate_caption_gpt2(gpt2_model, gpt2_tokenizer, image_tensor, device)
            tiny_caption = generate_caption_tiny(tiny_model, llama_tokenizer, image_tensor, device)

            results.append((image_path, target_caption, clip_caption, gpt2_caption, tiny_caption))

            print(f"✓ Target caption:   \"{target_caption}\"")
            print(f"✓ CLIP caption:     \"{clip_caption}\"")
            print(f"✓ GPT2 caption:     \"{gpt2_caption}\"")
            print(f"✓ TinyLlama caption:\"{tiny_caption}\"")
            print('-' * 50)

            if save_visualization:
                plt.subplot(len(image_paths), 1, idx + 1)
                img = Image.open(image_path)
                plt.imshow(img)
                plt.title(
                    f"Image: {os.path.basename(image_path)}\n" +
                    f"Target: \"{target_caption}\"\n" +
                    f"GPT2: \"{gpt2_caption}\"\n" +
                    f"TinyLlama: \"{tiny_caption}\"",
                    fontsize=10, pad=10
                )
                plt.axis('off')

        except Exception as e:
            print(f"✗ Error processing image: {str(e)}")
            print('-' * 50)

    if save_visualization and results:
        plt.tight_layout()
        plt.savefig('tinyllama_vs_gpt2.png', bbox_inches='tight', dpi=300)
        print("\nVisualization saved as 'tinyllama_vs_gpt2.png'")

    return results

if __name__ == '__main__':
    image_names, _ = load_flickr_data()
    sample_paths = [os.path.join('flickr/flickr30k_images/flickr30k_images', p) for p in image_names[:3]]
    evaluate_on_images(sample_paths)
