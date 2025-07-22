# CLIP Image Search Demo

This demo integrates a small Node.js backend with a Next.js
frontend for text–to–image search using a CLIP model.

## Setup

For a quick start you can run `python stupbackend.py`. This script will generate
the vector store if it does not exist and then launch the inference backend.
After it is running you only need to start the Next.js frontend.

1. **Generate the vector store**

   Pre-compute image embeddings and save them to `backend/vector_store.json`:

   ```bash
   python3 vector_store.py
   ```

   The script loads the images listed in `my-app/public/image_paths.csv`,
   computes embeddings using the model from `best.pt` and stores them
   for later use.

2. **Start both backend and frontend**

   Once the vector store is available you can launch the entire demo with a
   single command:

   ```bash
   ./scripts/start.sh
   ```

   The script starts the Node.js inference server on port `8000` and then runs
   the Next.js frontend (available on `http://localhost:3000`). `BACKEND_URL` is
   automatically set to the local backend, so you can simply visit
   `http://localhost:3000/images` in your browser. Submitting text in the search
   bar will call `/api/search` which proxies to the backend to retrieve the best
   matching image.

## Running on HPC (e.g. NERSC Perlmutter)

When using an interactive compute node you will typically want to forward ports
so that the frontend and backend are reachable from your local machine.  One
approach is to open an SSH tunnel before launching the demo:

```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 <user>@perlmutter.nersc.gov
```

After connecting, generate the vector store and start the services as usual:

```bash
python3 vector_store.py        # only needed once
BACKEND_URL=http://localhost:8000 ./scripts/start.sh
```

You can then browse to `http://localhost:3000/images` on your local machine.
If you encounter connection errors, verify that both ports are listening using:

```bash
python3 scripts/check_ports.py
```

The script will report whether the backend and frontend are reachable on the
expected ports.
