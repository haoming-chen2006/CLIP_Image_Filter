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

2. **Start the inference backend**

   ```bash
   node backend/index.js
   ```

   The server listens on `PORT` (default `8000`).

   The server loads the vector store on start and exposes two endpoints:

   - `GET /init` – loads embeddings if not yet loaded
   - `GET /search?q=<text>` – returns the filename of the best matching image

3. **Run the Next.js frontend**

   Set `BACKEND_URL` to the address of the inference server if it is not running
   on the same machine:

   ```bash
   export BACKEND_URL=http://<backend-host>:8000
   cd my-app
   npm run dev
   ```

   Visiting `http://localhost:3000/images` will trigger `/api/init` to load the
   embeddings. Submitting text in the search bar calls `/api/search` which
   proxies to the backend using `BACKEND_URL` to obtain the best matching image.
