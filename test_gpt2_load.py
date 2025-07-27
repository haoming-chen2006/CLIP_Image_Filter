import torch
from modules import GPT

def test_loading():
    """
    Tests if the GPT.from_pretrained method works correctly.
    """
    print("--- Starting GPT-2 Pretrained Model Loading Test ---")
    try:
        # Attempt to load the pretrained 'gpt2' model
        model = GPT.from_pretrained('gpt2')
        
        print("\n--- Test Summary ---")
        print("✅ Successfully loaded pretrained GPT-2 model.")
        print(f"   - Model class: {type(model)}")
        print(f"   - Number of parameters: {model.get_num_params()/1e6:.2f}M")
        
        # Optional: check if the model can do a forward pass
        print("\nTesting a simple forward pass...")
        # Create a dummy input tensor (batch size 1, sequence length 10)
        # vocab_size is 50257 for gpt2, so tokens should be in that range
        dummy_input = torch.randint(0, 50257, (1, 10)) 
        logits, loss = model(dummy_input)
        print(f"   - Forward pass successful.")
        print(f"   - Output logits shape: {logits.shape}")
        print("--- Test Finished ---")

    except Exception as e:
        print("\n--- Test Summary ---")
        print(f"❌ Test Failed: An error occurred during model loading.")
        print(f"   - Error: {e}")
        print("--- Test Finished ---")

if __name__ == "__main__":
    test_loading()
