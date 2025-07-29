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



image_names, captions = load_flickr_data()

