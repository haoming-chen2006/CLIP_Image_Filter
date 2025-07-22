import json
import torch
import torch.nn.functional as F
from inference import load_flickr_data, get_image_embeddings
from config import CFG

if __name__ == '__main__':
    image_names, comments = load_flickr_data()
    CFG.image_path = 'my-app/public/images'
    model_path = 'best.pt'
    model, image_embeddings, subset_filenames = get_image_embeddings(image_names, comments, model_path)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1).cpu().tolist()
    data = {
        'filenames': subset_filenames,
        'embeddings': image_embeddings
    }
    with open('backend/vector_store.json', 'w') as f:
        json.dump(data, f)
    print(f'Saved {len(subset_filenames)} embeddings to backend/vector_store.json')
