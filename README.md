# 🧠 Knowledge Graph RAG

> A Retrieval-Augmented Generation (RAG) system powered by **Neo4J Knowledge Graphs** and **LangChain** — combining structured graph reasoning with LLM-based question answering.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Neo4J](https://img.shields.io/badge/Neo4J-5.x-green)
![LangChain](https://img.shields.io/badge/LangChain-latest-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 Motivation

Traditional RAG systems retrieve flat text chunks from vector databases. This project explores a **graph-enhanced approach**: entities and their relationships are stored in a Neo4J Knowledge Graph, enabling the LLM to reason over **structured connections** — not just similar text.

This is particularly powerful for domains where relationships matter: education, healthcare, e-commerce, and enterprise knowledge management.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Query Processor   │  → Entity extraction (spaCy / LLM)
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌────────┐  ┌──────────────┐
│ Neo4J  │  │ Vector Store │  (Chroma / Qdrant)
│  KG    │  │  Embeddings  │
└────┬───┘  └──────┬───────┘
     │              │
     └──────┬───────┘
            │  Context Fusion
            ▼
    ┌───────────────┐
    │  LLM (Claude  │
    │  / GPT-4)     │
    └───────┬───────┘
            │
            ▼
       Final Answer
```

---

## ✨ Features

- 🔗 **Graph-based retrieval** via Neo4J Cypher queries
- 🧩 **Hybrid search** — combines graph traversal + semantic vector search
- 🤖 **LangChain integration** with `GraphCypherQAChain`
- 📊 **Knowledge Graph builder** — ingest raw text and extract entities/relations automatically
- 🖥️ **FastAPI endpoint** for easy integration
- 📓 **Jupyter notebooks** for exploration and demos

---

## Screenshot

![Application Screenshot](https://github.com/Arthuro1/knowledge-graph-rag/blob/main/images/ui.PNG?raw=true)

---

## 🚀 Quickstart

### Prerequisites

- Python 3.11+
- Neo4J Desktop or Docker
- OpenAI or Anthropic API key

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/knowledge-graph-rag.git
cd knowledge-graph-rag
pip install -r requirements.txt
```

### 2. Start Neo4J

```bash
docker-compose up -d
```

Neo4J Browser: http://localhost:7474 (user: `neo4j`, password: `password`)

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and Neo4J credentials
```

### 4. Build the Knowledge Graph

```bash
python src/graph/kg_builder.py --input data/raw/sample_docs.txt
```

### 5. Run a query

```bash
python src/main.py --query "What are the prerequisites for machine learning?"
```

### 6. Start the Web-UI

```bash
uvicorn src.api.app:app --reload
# Web-UI: http://localhost:8000
```

### 7. Start the API

```bash
uvicorn src.api.app:app --reload
# API docs: http://localhost:8000/docs
```

---

## 📁 Project Structure

```
knowledge-graph-rag/
├── data/
│   ├── raw/                  # Input documents (txt, pdf, json)
│   └── processed/            # Extracted entities & relations
├── src/
│   ├── graph/
│   │   ├── kg_builder.py     # Entity/relation extraction → Neo4J
│   │   ├── neo4j_client.py   # Neo4J connection & Cypher queries
│   │   └── schema.py         # Graph schema definition
│   ├── rag/
│   │   ├── retriever.py      # Hybrid graph + vector retriever
│   │   ├── embeddings.py     # Embedding model wrapper
│   │   └── chain.py          # LangChain RAG chain
│   ├── api/
│   │   └── app.py            # FastAPI application
│   └── main.py               # CLI entrypoint
├── notebooks/
│   ├── 01_kg_exploration.ipynb    # Explore the knowledge graph
│   ├── 02_rag_pipeline.ipynb      # RAG pipeline walkthrough
│   └── 03_evaluation.ipynb        # Evaluate retrieval quality
├── tests/
│   ├── test_kg_builder.py
│   ├── test_retriever.py
│   └── test_api.py
├── docs/
│   └── architecture.md
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧪 Example

```python
from src.rag.chain import GraphRAGChain

chain = GraphRAGChain()
response = chain.query("Which topics are related to neural networks?")
print(response)
# → "Neural networks are related to: deep learning, backpropagation,
#    activation functions, and gradient descent. In the knowledge graph,
#    they connect directly to CNNs, RNNs, and Transformers..."
```

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `01_kg_exploration.ipynb` | Visualize the graph, run Cypher queries |
| `02_rag_pipeline.ipynb` | End-to-end RAG demo step by step |
| `03_evaluation.ipynb` | Compare graph RAG vs. standard RAG |

---

## 🛣️ Roadmap

- [ ] Support PDF ingestion
- [ ] Add GraphRAG (Microsoft) comparison
- [ ] Streamlit demo UI
- [ ] Multi-hop reasoning over graph paths
- [ ] Evaluation with RAGAS framework

---

## 🤝 Related Work

- [DeepLearning.AI – Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/)
- [LangChain GraphCypherQAChain](https://python.langchain.com/docs/use_cases/graph/graph_cypher_qa_chain)
- [LightRAG – Knowledge Graph + RAG](https://github.com/HKUDS/LightRAG)

---

## 👤 Author

**Paul Meteng** — Software Engineer & AI Engineer  
[LinkedIn](https://www.linkedin.com/in/paul-arthur-meteng-kamdem-06b922125/) · [GitHub](https://github.com/Arthuro1)

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
