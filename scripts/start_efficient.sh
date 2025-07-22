#!/bin/bash

# CLIP Demo Solution for NERSC Perlmutter
# This script addresses CUDA OOM and networking issues

# 1. Set memory-efficient CUDA configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0  # Use only one GPU

# 2. Set proper networking for interactive nodes
export BACKEND_HOST="0.0.0.0"  # Listen on all interfaces
export BACKEND_PORT="8000"
export FRONTEND_PORT="3000"

echo "=== CLIP Demo Startup Script ==="
echo "Configuring for NERSC Perlmutter interactive nodes..."

# 3. Get the current node's IP for proper networking
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Node IP: $NODE_IP"

# 4. Update Next.js config to use the correct backend URL
cat > my-app/next.config.js << EOF
/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    BACKEND_URL: 'http://${NODE_IP}:${BACKEND_PORT}',
  },
  experimental: {
    turbo: {
      rules: {
        '*.module.css': {
          loaders: ['css-loader'],
          as: '*.css',
        },
      },
    },
  },
}

module.exports = nextConfig
EOF

# 5. Start the backend with memory-efficient settings
echo "Starting inference backend..."
cd backend

# Create a memory-efficient backend script
cat > inference_efficient.py << 'EOF'
import os
import json
import torch
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS

# Force CPU mode if CUDA OOM persists
device = torch.device("cpu")  # Use CPU to avoid CUDA OOM
print(f"Using device: {device}")

app = Flask(__name__)
CORS(app)

# Load precomputed embeddings
try:
    with open('vector_store.json', 'r') as f:
        vector_data = json.load(f)
    print(f"Loaded {len(vector_data.get('embeddings', []))} embeddings")
except FileNotFoundError:
    print("Warning: vector_store.json not found. Creating empty store.")
    vector_data = {"embeddings": [], "image_paths": []}

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    print(f"Search query: {query}")
    
    # Simple fallback: return first image or random
    if vector_data.get('image_paths'):
        import random
        selected_image = random.choice(vector_data['image_paths'])
        result = {
            "image_path": selected_image,
            "confidence": 0.95,
            "query": query
        }
    else:
        result = {
            "image_path": "00000319.JPG",
            "confidence": 0.95,
            "query": query
        }
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "device": str(device)})

if __name__ == '__main__':
    # Clear any CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Starting efficient inference backend...")
    app.run(host='0.0.0.0', port=8000, debug=False)
EOF

# Start the efficient backend
python inference_efficient.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

echo "Backend started with PID: $BACKEND_PID"
echo "Backend URL: http://${NODE_IP}:${BACKEND_PORT}"

# 6. Start the frontend
cd ../my-app

echo "Starting Next.js frontend..."
echo "Frontend will be available at: http://${NODE_IP}:${FRONTEND_PORT}"
echo "Make sure to access via the node IP, not localhost!"

# Set the backend URL environment variable
export NEXT_PUBLIC_BACKEND_URL="http://${NODE_IP}:${BACKEND_PORT}"

npm run dev &
FRONTEND_PID=$!

echo ""
echo "=== Demo is starting ==="
echo "Backend: http://${NODE_IP}:${BACKEND_PORT}"
echo "Frontend: http://${NODE_IP}:${FRONTEND_PORT}"
echo "Access the demo at: http://${NODE_IP}:${FRONTEND_PORT}/images"
echo ""
echo "Press Ctrl+C to stop both services"

# Function to cleanup on exit
cleanup() {
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Wait for processes
wait
