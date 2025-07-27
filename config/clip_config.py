import os
import torch


class ClipConfig:
    debug = False
    dataset_root = "/pscratch/sd/h/haoming/Projects/clip/flickr/flickr30k_images"
    image_path = '/pscratch/sd/h/haoming/Projects/clip/flickr/flickr30k_images/flickr30k_images'
    captions_path = dataset_root
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
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

    # Image encoder settings
    model_name = 'resnet50'
    image_embedding = 2048

    # Text encoder settings
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    # Model settings
    pretrained = True
    trainable = True
    temperature = 1.0

    # Image preprocessing
    size = 256

    # Projection head settings
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1
