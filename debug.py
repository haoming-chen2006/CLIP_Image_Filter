import os
import requests

print("*** HuggingFace Connectivity Debugger ***")

# Check PyTorch and CUDA
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
except Exception as e:
    print(f"PyTorch not installed or CUDA check failed: {e}")

# Check network access to huggingface.co
try:
    r = requests.get("https://huggingface.co", timeout=5)
    if r.status_code == 200:
        print("Connectivity to huggingface.co: OK")
    else:
        print(f"Connectivity to huggingface.co: Status code {r.status_code}")
except Exception as e:
    print(f"Failed to reach huggingface.co: {e}")

# Check huggingface_hub availability and login status
try:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    try:
        who = api.whoami()
        name = who.get('name', who)
        print(f"Logged in to Hugging Face as: {name}")
    except Exception as e:
        print(f"No Hugging Face login found or login check failed: {e}")
    try:
        hf_hub_download(repo_id="gpt2", filename="config.json", cache_dir=".hf-debug-cache")
        print("Able to download gpt2 config from Hugging Face")
    except Exception as e:
        print(f"Failed to download gpt2 config: {e}")
except Exception as e:
    print(f"huggingface_hub not installed: {e}")

# Check HF_TOKEN env variable
if os.environ.get("HF_TOKEN"):
    print("HF_TOKEN environment variable is set")
else:
    print("HF_TOKEN environment variable not set")
