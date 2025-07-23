# CLIP Image Search Demo

I am very unorganized when it comes to image searching and lazy when it comes to editing, so I find it useful to have a semantic image to test search tool that can help me to find the right image in my album and an editor to make the image look better. I used custom-trained CLIP (Contrastive Language-Image Pre-training) model to enable natural language queries for image retrieval and used a CLIP-guided diffusion model for image editing. This demo features a Node.js backend for inference and a Next.js frontend for an intuitive web interface.

## ✨ Capabilities

### Primary Feature: Semantic Image Search
- **Natural Language Queries**: Search using descriptive text in plain English
- **Semantic Understanding**: Finds conceptually similar images, wtihout the need of captions
- **Real-time Inference**: Fast CPU-based search after initial embedding computation

### Secondary Feature: CLIP-Guided Image Enhancement & Generation
- **Diffusion-Based Image Editing**: Uses guided diffusion to enhance existing scenery images through denoising and lighting improvements
- **Intelligent Denoising**: CLIP guidance ensures semantic preservation while removing noise and artifacts
- **Dynamic Lighting Enhancement**: Adjusts exposure, contrast, and color temperature based on scene understanding

## 🏗️ Model Architecture

The system implements a dual-encoder CLIP architecture with the following components:

### Image Encoder
- **Base Model**: ResNet-50 (via TIMM) with pretrained ImageNet weights
- **Input Resolution**: 256×256 pixels
- **Output Embedding**: 2048-dimensional feature vector
- **Custom Architecture**: Enhanced with RMSNorm, SiLU activations, and residual connections
- **Alternative**: Vision Transformer (ViT) encoder with patch embedding and multi-head attention

### Text Encoder  
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Tokenization**: Handles up to 200 tokens per query
- **Output Embedding**: 768-dimensional feature vector from CLS token
- **Architecture**: Transformer-based with 6 layers, 12 attention heads

### Projection Heads
- **Shared Dimensionality**: Both encoders project to 256-dimensional space
- **Architecture**: Linear projection → GELU → Linear → Dropout → Residual connection → LayerNorm
- **Purpose**: Creates common embedding space for similarity computation

### Training Configuration
- **Contrastive Loss**: Uses temperature scaling (τ=1.0) for stable training
- **Optimization**: AdamW with differential learning rates:
  - Image encoder: 1e-4
  - Text encoder: 1e-5  
  - Projection heads: 1e-3
- **Regularization**: Weight decay (1e-3), dropout (0.1)
- **Hardware Support**: CUDA, Apple MPS, or CPU fallback

## 📊 Datasets

### Flickr30K Dataset
The model was trained on the Flickr30K dataset, containing:
- **Images**: 31,783 images from Flickr
- **Captions**: ~158K human-annotated captions (5 per image)
- **Content**: Diverse scenes including people, animals, objects, and activities
- **Quality**: Professional and amateur photography with rich descriptive text

**Sample Image-Caption Pairs:**

![Flickr30K Dataset Samples](flickr_samples.png)
*Sample images from the Flickr30K dataset showing diverse scenes with their corresponding captions: climbing scenes, outdoor activities, social gatherings, and more.*

### Instagram Dataset (Custom Collection)
Additionally trained on curated Instagram images featuring:
- **Images**: Personal photography collection (IMG_*.JPG files)
- **Diversity**: Urban scenes, portraits, nature, and lifestyle photography  
- **Style**: Modern social media aesthetic with varied compositions
- **Enhancement**: Expands model's understanding of contemporary visual culture

### Scenery & City Datasets (Kaggle)
For the CLIP-guided enhancement feature, additional training was performed on specialized datasets:

#### Natural Scenery Dataset
- **Source**: Kaggle Natural Landscapes Collection
- **Images**: ~25,000 high-quality landscape photographs
- **Categories**: Mountains, forests, beaches, deserts, rivers, and countryside
- **Lighting Conditions**: Dawn, dusk, golden hour, overcast, and clear weather
- **Quality Focus**: Professional nature photography with excellent lighting and composition
- **Enhancement Training**: Paired low/high quality images for learning denoising and lighting improvements

#### Urban Cityscapes Dataset  
- **Source**: Kaggle City Scenes & Architecture Dataset
- **Images**: ~15,000 urban photography samples
- **Categories**: Skylines, street scenes, architectural details, public spaces
- **Time Variations**: Day/night cycles, different weather conditions, seasonal changes
- **Style Range**: Modern cities, historical architecture, industrial areas
- **Technical Focus**: Trained on enhancing urban lighting, reducing noise from night photography, and improving atmospheric conditions

## ✨ Capabilities

### Primary Feature: Semantic Image Search
- **Natural Language Queries**: Search using descriptive text in plain English
- **Semantic Understanding**: Finds conceptually similar images, wtihout the need of captions
- **Real-time Inference**: Fast CPU-based search after initial embedding computation

### Secondary Feature: CLIP-Guided Image Enhancement & Generation
- **Diffusion-Based Image Editing**: Uses guided diffusion to enhance existing scenery images through denoising and lighting improvements
- **Intelligent Denoising**: CLIP guidance ensures semantic preservation while removing noise and artifacts
- **Dynamic Lighting Enhancement**: Adjusts exposure, contrast, and color temperature based on scene understanding



![Web Application Interface](Screenshot%202025-07-22%20at%2014.52.07.png)
*The web interface demonstrates a clean, modern design with a prominent search bar and image results displayed in an organized grid layout, showing real-time text-to-image search capabilities.*

## 🚀 Quick Start

For immediate setup, run the automated backend script:
```bash
python stupbackend.py
```
This will generate embeddings if needed and launch the inference server.
```

This implementation demonstrates the practical application of multimodal AI for both content discovery and creative generation, providing a foundation for building sophisticated image search and synthesis systems.
