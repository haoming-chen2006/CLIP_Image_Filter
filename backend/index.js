const http = require('http');
const fs = require('fs');
const { spawn } = require('child_process');
const path = require('path');

const PORT = process.env.PORT || 8000;
let vectorStore = null;

function loadVectorStore() {
  const file = path.join(__dirname, 'vector_store.json');
  if (!fs.existsSync(file)) {
    console.error('Vector store not found:', file);
    return;
  }
  const data = JSON.parse(fs.readFileSync(file, 'utf-8'));
  vectorStore = data;
  console.log(`Loaded ${data.filenames.length} image embeddings`);
}

function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function handleSearch(query, res) {
  if (!vectorStore) {
    loadVectorStore();
    if (!vectorStore) {
      res.writeHead(500, {'Content-Type':'application/json'});
      res.end(JSON.stringify({error:'Vector store not loaded'}));
      return;
    }
  }
  const py = spawn('python3', [path.join(__dirname, '..', 'inference.py'), '--encode', query]);
  let output = '';
  py.stdout.on('data', (d) => { output += d.toString(); });
  py.stderr.on('data', (d) => { console.error(d.toString()); });
  py.on('close', (code) => {
    if (code !== 0) {
      res.writeHead(500, {'Content-Type':'application/json'});
      res.end(JSON.stringify({error:'Python error'}));
      return;
    }
    try {
      const result = JSON.parse(output.trim());
      const queryVec = result.embedding;
      let bestScore = -Infinity;
      let bestIndex = -1;
      for (let i = 0; i < vectorStore.embeddings.length; i++) {
        const score = cosineSimilarity(queryVec, vectorStore.embeddings[i]);
        if (score > bestScore) {
          bestScore = score;
          bestIndex = i;
        }
      }
      const image = vectorStore.filenames[bestIndex];
      res.writeHead(200, {'Content-Type':'application/json'});
      res.end(JSON.stringify({ image }));
    } catch (e) {
      console.error('Failed to parse python output', e);
      res.writeHead(500, {'Content-Type':'application/json'});
      res.end(JSON.stringify({error:'Invalid output'}));
    }
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  if (url.pathname === '/search' && url.searchParams.has('q')) {
    const query = url.searchParams.get('q');
    handleSearch(query, res);
  } else if (url.pathname === '/init') {
    loadVectorStore();
    res.writeHead(200, {'Content-Type':'application/json'});
    if (vectorStore) {
      res.end(JSON.stringify({status:'ok', count: vectorStore.filenames.length}));
    } else {
      res.end(JSON.stringify({status:'failed'}));
    }
  } else {
    res.writeHead(404, {'Content-Type':'application/json'});
    res.end(JSON.stringify({error:'Not found'}));
  }
});

server.listen(PORT, () => {
  console.log(`Inference backend running on http://localhost:${PORT}`);
  loadVectorStore();
});
