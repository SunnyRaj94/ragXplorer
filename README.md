# 🧠 RAGXPLORER

**RAGXPLORER** is a hands-on, modular playground for mastering the Retrieval-Augmented Generation (RAG) stack. Designed for machine learning engineers, it provides notebooks and utilities to explore LLMs, embeddings, vector DBs, document parsing, and evaluation tools—all in one developer-friendly repo.

![Logo](./assets/logo.png)

---

## 🚀 Project Goals

- Explore RAG components using real tools (LangChain, Chroma, Ollama, FireCrawl, RAGAS, etc.)
- Provide clean, Jupyter-based walkthroughs for each part of the stack
- Package core logic in a reusable Python library (`rag_utils`)
- Serve as a reference stack for local or cloud-deployed RAG pipelines

---

## 📁 Structure

```

ragverse/
├── notebooks/           # Organized by RAG stages
├── rag\_utils/           # Reusable Python library
├── tests/               # Test scripts for utilities
├── requirements.txt     # Dependency list
└── README.md

````

---

## 🔍 Components Covered

| Category        | Tools Included |
|----------------|----------------|
| LLMs           | OpenAI, Ollama, Mistral, Claude |
| Frameworks     | LangChain, LlamaIndex, Haystack |
| Vector DBs     | Chroma, Qdrant, Weaviate |
| Data Extraction| LlamaParse, FireCrawl, Docling |
| Embeddings     | SBERT, OpenAI, BGE, Nomic |
| Evaluation     | RAGAS, Giskard |

---

## 🧪 Getting Started

```bash
git clone https://github.com/your-username/ragexplorer.git
cd ragexplorer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

Then open the Jupyter notebooks:

```bash
jupyter lab
```

---

### 🔧 Local Setup

#### 🐍 Create & activate a virtual environment (optional)

```bash
python -m venv rag_env
source rag_env/bin/activate  # or .\rag_env\Scripts\activate on Windows
```

#### 📦 Install dependencies

```bash
pip install -e .
```

#### 🧠 Running Local LLMs with Ollama

This project uses [Ollama](https://ollama.com) to run local LLMs like Mistral or LLaMA 3.

##### ✅ Install Ollama on macOS

```bash
brew install ollama
```

If you don’t have Homebrew, install it first from: [https://brew.sh](https://brew.sh)

##### 🛠️ Start the Ollama server

```bash
ollama serve
```

##### 📥 Pull or run a model

```bash
ollama run mistral
# or just pull the model without running
ollama pull mistral
```

##### 🧪 Test the API

```bash
curl http://localhost:11434/api/tags
```

You should see a list of available models.

---

## 🛠️ Example Notebooks

* `01_ingestion/web_scraping_firecrawl.ipynb`
* `02_indexing/chroma_indexing.ipynb`
* `04_llms/openai_integration.ipynb`
* `07_end_to_end/mini_rag_pipeline.ipynb`

---

## 🧠 Author

Made with 💡 by Sunny Raj — feel free to contribute, fork, or raise issues.

---

## 📜 License

MIT License. Free to use, learn, and build with.