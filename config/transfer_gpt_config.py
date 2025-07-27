import os
import torch


class TransferGPTConfig:
    debug = False
    dataset_root = "/pscratch/sd/h/haoming/Projects/clip/flickr/flickr30k_images"
    image_path = '/pscratch/sd/h/haoming/Projects/clip/flickr/flickr30k_images/flickr30k_images'
    captions_path = dataset_root
    batch_size = 32
    num_workers = 4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 20

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Model paths
    clip_model_path = "best.pt"  # Path to the trained CLIP model

    # Image preprocessing
    size = 256

    # TinyLlama settings (or any other model you want to use)
    gpt_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length = 50
    temperature = 1.0
    top_k = 50

    # Training settings
    learning_rate = 1e-4
    train_batch_size = 32
    eval_batch_size = 32
    projection_dim = 256  # Should match CLIP's projection_dim
