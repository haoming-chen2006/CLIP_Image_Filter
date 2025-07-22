# CLIP Image Search Demo

I am very unorganized when it comes to image searching, so I find it useful to have a semantic image to test search tool that can help me to find the right image in my album. I am trying to build a tool that is very simple to use and connect to album I used custom-trained CLIP (Contrastive Language-Image Pre-training) model to enable natural language queries for image retrieval. This demo features a Node.js backend for inference and a Next.js frontend for an intuitive web interface.

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
```
Image: 00000005.JPG
Caption: "A person in a yellow jacket climbing a steep rock face"

Image: 00000027.JPG  
Caption: "Two dogs running through a grassy field with trees in the background"

Image: 00000104.JPG
Caption: "A group of people sitting around a table in a restaurant"
```

### Instagram Dataset (Custom Collection)
Additionally trained on curated Instagram images featuring:
- **Images**: Personal photography collection (IMG_*.JPG files)
- **Diversity**: Urban scenes, portraits, nature, and lifestyle photography  
- **Style**: Modern social media aesthetic with varied compositions
- **Enhancement**: Expands model's understanding of contemporary visual culture

## ✨ Capabilities

### Core Features
- **Natural Language Queries**: Search using descriptive text in plain English
- **Semantic Understanding**: Finds conceptually similar images beyond exact keyword matches
- **Real-time Inference**: Fast CPU-based search after initial embedding computation
- **Scalable Architecture**: Handles large image collections efficiently through vector similarity

### Advanced Search Examples
The system excels at understanding:
- **Objects & Entities**: "red car", "golden retriever", "mountain landscape"
- **Actions & Activities**: "people dancing", "children playing", "cooking food" 
- **Scenes & Contexts**: "beach sunset", "city street at night", "cozy interior"
- **Emotions & Moods**: "happy celebration", "peaceful nature", "dramatic lighting"
- **Artistic Qualities**: "black and white photo", "vibrant colors", "minimalist composition"

### Performance Metrics
- **Embedding Dimension**: 256D normalized vectors for efficient similarity computation
- **Search Speed**: Sub-second response times using cosine similarity
- **Memory Efficiency**: Pre-computed embeddings eliminate need for model inference during search
- **Accuracy**: High semantic relevance through contrastive learning on large-scale datasets

## 🖥️ Web Application

The demo includes a modern web interface built with Next.js and Tailwind CSS:

### Features Demonstrated
- **Interactive Search Bar**: Real-time text input with instant results
- **Image Gallery**: Grid-based display of search results
- **Responsive Design**: Optimized for desktop and mobile viewing
- **Fast Loading**: Efficient image serving from local storage
- **Error Handling**: Graceful fallbacks for connectivity issues

### User Interface
The web application provides an intuitive experience where users can:
1. Enter natural language descriptions in the search field
2. Instantly see the most relevant images from the dataset
3. Explore diverse visual content through semantic search
4. Experience the power of multimodal AI in a practical application

*Screenshot: The web interface shows a clean, modern design with a prominent search bar and image results displayed in an organized grid layout, demonstrating real-time text-to-image search capabilities.*

## 🚀 Quick Start

For immediate setup, run the automated backend script:
```bash
python stupbackend.py
```
This will generate embeddings if needed and launch the inference server.

## 📋 Manual Setup

### 1. Generate Vector Store
Pre-compute image embeddings for fast search:
```bash
python3 vector_store.py
```

### 2. Install Dependencies
Ensure all Python packages are installed:
```bash
pip install torch torchvision transformers timm albumentations opencv-python einops pandas matplotlib pillow tqdm
```

Install Node.js dependencies:
```bash
cd my-app && npm install
```

### 3. Launch Application
Start both backend and frontend:
```bash
./scripts/start.sh
```
- Backend runs on `http://localhost:8000` 
- Frontend available at `http://localhost:3000/images`

## 🏥 HPC Deployment (NERSC Perlmutter)

For high-performance computing environments:

```bash
# SSH tunnel for port forwarding
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 <user>@perlmutter.nersc.gov

# Generate embeddings and start services
python3 vector_store.py
BACKEND_URL=http://localhost:8000 ./scripts/start.sh
```

Verify connectivity:
```bash
python3 scripts/check_ports.py
```

## 🔧 Technical Details

### System Requirements
- **Python**: 3.8+ with PyTorch ecosystem
- **Node.js**: 16+ for backend and frontend services
- **Memory**: Minimum 4GB RAM for embedding computation
- **Storage**: ~500MB for model weights and image dataset

### Architecture Benefits
- **Modular Design**: Separate image/text encoders enable flexible training
- **Efficient Inference**: Vector similarity search scales to millions of images
- **Cross-Modal Understanding**: Joint embedding space enables text↔image retrieval
- **Production Ready**: Clean API separation between Python ML backend and Node.js web services

### File Structure
```
CLIP_Image_Filter/
├── model_architecture/     # Custom encoder implementations
├── backend/                # Node.js inference server  
├── my-app/                # Next.js web application
├── scripts/               # Deployment utilities
├── best.pt               # Trained model weights
├── vector_store.py       # Embedding pre-computation
└── inference.py          # Core search functionality
```

This implementation demonstrates the practical application of multimodal AI for content discovery, providing a foundation for building sophisticated image search systems.
