import torch
from transformers import AutoTokenizer
from dataset import CLIPDataset, get_transforms, load_flickr_data
from train_transfer_caption_gpt2 import CLIPTransferCaptionModelGPT2
from config import CFG

def test_single_batch():
    print("Loading data...")
    image_names, captions = load_flickr_data()
    
    # Take just a few samples
    image_names = image_names[:4]
    captions = captions[:4]
    
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nCreating dataset...")
    dataset = CLIPDataset(image_names, captions, tokenizer, get_transforms("train"))
    
    print("\nCreating dataloader...")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,  # Use 0 for debugging
        shuffle=False
    )
    
    print("\nInitializing model...")
    model = CLIPTransferCaptionModelGPT2(gpt_name="gpt2").to(CFG.device)
    model.eval()
    
    print("\nTesting forward pass...")
    for batch in loader:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        with torch.no_grad():
            loss = model(batch)
        print("Loss:", loss.item())
        break
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_single_batch()
