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
from inference import load_flickr_data, get_image_embeddings

image_names,comments = load_flickr_data(csv_path = "/pscratch/sd/h/haoming/Projects/clip/flickr/results.csv")
CFG.image_path = "/pscratch/sd/h/haoming/Projects/clip/flickr/flickr30k_images/flickr30k_images"
print(f"Loaded {len(image_names)} images")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    print("Please train the model first using train.py")
    
    # Precompute image embeddings
print("Computing image embeddings...")
model, image_embeddings, subset_filenames = get_image_embeddings(image_names, comments, model_path)

print(image_embeddings[:100])

