import os
import subprocess
import sys

VECTOR_STORE = os.path.join('backend', 'vector_store.json')


def ensure_vector_store():
    """Generate the vector store if it doesn't exist."""
    if os.path.exists(VECTOR_STORE):
        print('Vector store already exists. Skipping generation.')
        return
    print('Vector store not found. Generating embeddings...')
    result = subprocess.run([sys.executable, 'vector_store.py'])
    if result.returncode != 0:
        raise RuntimeError('Failed to generate vector store')


def start_backend():
    """Start the Node.js inference server."""
    print('Starting inference backend...')
    proc = subprocess.Popen(['node', os.path.join('backend', 'index.js')])
    try:
        proc.wait()
    except KeyboardInterrupt:
        print('Stopping backend...')
        proc.terminate()
        proc.wait()


def main():
    ensure_vector_store()
    start_backend()


if __name__ == '__main__':
    main()
