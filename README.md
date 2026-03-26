# 🧠 Edge AI Knowledge Assistant
### Offline RAG System — Mac Development → Jetson Nano Deployment

> Upload documents. Ask questions. Get accurate answers.  
> **Completely offline. No cloud. No API keys. No data leakage.**

---

## What Is This?

A fully offline **Retrieval-Augmented Generation (RAG)** system that lets you query your own PDF documents in natural language — without sending anything to the internet.

Built for edge deployment on **NVIDIA Jetson Nano**, with a Mac-first development workflow so you can build and test everything locally before deploying to hardware.

**The core idea:** Instead of using ChatGPT's general knowledge, this system reads *your* documents, finds the most relevant sections, and generates answers grounded only in that content — all on your own machine.

---

## How It Works

```
PDF Upload → Text Extraction → Chunking → Embedding → FAISS Store
                                                            ↓
User Question → Embed Question → Vector Search → Top-K Chunks
                                                            ↓
                                          Local LLM → Grounded Answer
```

1. **Ingest** — PDFs are parsed, split into 500-character chunks, and stored with metadata (filename, page number)
2. **Embed** — Each chunk is converted into a 384-dim vector using `all-MiniLM-L6-v2`
3. **Store** — Vectors are indexed in FAISS locally (no server required)
4. **Query** — Your question is embedded, top-3 relevant chunks are retrieved
5. **Generate** — A local LLM reads only those chunks and produces a grounded answer with source citations

---

## Tech Stack

| Component | Mac (Dev) | Jetson Nano (Deploy) |
|---|---|---|
| **PDF Parsing** | PyMuPDF (fitz) | PyMuPDF (fitz) |
| **Text Chunking** | LangChain RecursiveTextSplitter | Same |
| **Embeddings** | all-MiniLM-L6-v2 (80MB) | Same |
| **Vector DB** | FAISS CPU | FAISS CPU |
| **Local LLM** | Ollama + TinyLlama | llama.cpp (Q4_K_M) |
| **UI** | Streamlit | Streamlit |

> **Why two LLM setups?** Ollama doesn't support Jetson Nano's ARM64 + Maxwell GPU architecture. On Jetson, we use `llama.cpp` built directly with CUDA support. Everything else stays identical.

---

## Project Structure

```
edge-rag/
├── ingestion/
│   ├── parser.py          # PDF → pages with metadata
│   └── chunker.py         # Pages → overlapping chunks
├── embeddings/
│   └── embedder.py        # Text → 384-dim vectors (MiniLM)
├── vectorstore/
│   └── store.py           # FAISS index: add / search / save / load
├── llm/
│   └── local_llm.py       # Ollama wrapper (swap for llama.cpp on Jetson)
├── rag/
│   └── pipeline.py        # Full RAG chain: embed → retrieve → generate
├── eval/
│   └── measure.py         # Latency, RAM, GPU benchmarking
├── app.py                 # Streamlit UI
└── requirements.txt
```

---

## Quickstart (Mac)

### 1. Clone & set up environment

```bash
git clone https://github.com/your-username/edge-rag.git
cd edge-rag

python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install streamlit pymupdf sentence-transformers faiss-cpu langchain langchain-community ollama
```

### 3. Install Ollama and pull the model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull TinyLlama (~700MB, one-time download)
ollama pull tinyllama
```

### 4. Run the app

```bash
# Terminal 1 — keep Ollama running
ollama serve

# Terminal 2 — launch the UI
streamlit run app.py
```

Open your browser at `http://localhost:8501`, upload a PDF, and start asking questions.

---

## Usage

1. **Upload PDFs** using the sidebar file uploader
2. Click **Index Documents** — wait for the spinner to finish
3. Type a question in the main panel
4. Click **Ask** — answer appears with page-level source citations
5. Expand **"See retrieved context"** to see exactly which chunks the LLM used

The vector index is saved to `vectorstore.pkl` so it persists across sessions. Use **Clear Index** to reset.

---

## Jetson Nano Deployment

> Build and test everything on Mac first. Only swap `local_llm.py` for Jetson.

### Prerequisites
- JetPack 4.6.x (Ubuntu 18.04)
- Python 3.6+
- 4GB RAM Jetson Nano

### Build llama.cpp on Jetson

```bash
# Install build tools
sudo apt update && sudo apt install cmake build-essential python3-pip git -y

# Clone and build with CUDA support
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON    # enables Jetson Maxwell GPU
make -j4

# Install Python bindings
pip3 install llama-cpp-python
```

### Download a quantised model

```bash
mkdir -p models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -O models/tinyllama.gguf
```

### Swap the LLM wrapper

Replace `llm/local_llm.py` with:

```python
from llama_cpp import Llama

class LocalLLM:
    def __init__(self, model_path="models/tinyllama.gguf"):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=20,   # offload to Jetson GPU
            n_threads=4,       # Jetson has 4 CPU cores
            verbose=False
        )

    def generate(self, prompt: str) -> str:
        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            stop=["</s>", "User:", "Question:"]
        )
        return output["choices"][0]["text"].strip()
```

Everything else — RAG pipeline, embeddings, FAISS, Streamlit — stays exactly the same.

---

## Performance Targets (Jetson Nano)

| Metric | Target | Tool |
|---|---|---|
| Response latency | < 60 seconds | `time.time()` |
| RAM usage | < 3.5 GB | `psutil` |
| Power consumption | Measured | `tegrastats` |
| Answer accuracy | Manual eval | Test question set |
| Retrieval precision | Top-3 chunk relevance | Manual inspection |

Run benchmarks with:

```bash
python eval/measure.py
```

---

## Domain Applications

This system can be specialised for any document-heavy domain:

- **Manufacturing** — Machine SOPs, troubleshooting guides, safety manuals
- **Healthcare** — Hospital protocols, internal guidelines (non-diagnostic)
- **Legal** — Compliance manuals, company policies, government regulations
- **Education** — Course material, college handbooks, campus knowledge bases
- **Enterprise** — HR documents, onboarding guides, internal knowledge bases

---

## Why This Project Exists

Most RAG systems assume cloud infrastructure or high-end hardware. This project explores what's possible on a **$99 edge device** — useful for:

- Environments where internet is restricted or unreliable
- Deployments where documents are too sensitive to send to the cloud
- Cost-sensitive use cases that can't afford API bills
- Academic research into edge AI optimisation

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ollama: command not found` | Re-run the Ollama install curl command |
| `ModuleNotFoundError` | Ensure `(venv)` is active, re-run `pip install` |
| `Connection refused` | Run `ollama serve` in a separate terminal tab |
| Slow responses | Expected — TinyLlama takes 10–20s on Mac CPU |
| Poor answer quality | Try `ollama pull mistral` and update model name in `local_llm.py` |
| Jetson: out of memory | Reduce `n_gpu_layers` in llama.cpp config, or use a smaller model |

---

## Roadmap

- [x] Core RAG pipeline (Mac)
- [x] Streamlit UI with multi-PDF support
- [x] Persistent vector store
- [ ] Jetson Nano deployment with llama.cpp
- [ ] Hardware benchmarking script
- [ ] Role-based access control
- [ ] Multi-device sync
- [ ] Domain-specific fine-tuning support

---

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Local LLM inference
- [Ollama](https://ollama.com) — Mac LLM runtime
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` embeddings
- [LangChain](https://langchain.com) — Text chunking utilities
- [Streamlit](https://streamlit.io) — UI framework

---

*Capstone Project — Edge AI + RAG + Jetson Nano*

