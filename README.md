# RAG

```

rag-collection/
│── README.md                  # Overview of repo + explanations
│── requirements.txt           # Common dependencies
│── utils/                     # Shared code: embedding utils, vector DB wrapper, evaluation metrics
│
├── 01-vanilla-rag/            # Classic retrieval + generation
│   ├── app.py
│   ├── retriever.py
│   └── README.md
│
├── 02-rag-with-reranker/      # Retriever + reranker
│   ├── app.py
│   └── README.md
│
├── 03-conversational-rag/     # Multi-turn chat + query rewriting
│   ├── app.py
│   └── README.md
│
├── 04-self-rag/               # LLM decides retrieval strategy
│   ├── app.py
│   └── README.md
│
├── 05-hyde/                   # Hypothetical document embeddings
│   ├── app.py
│   └── README.md
│
├── 06-fid/                    # Fusion-in-decoder style
│   ├── app.py
│   └── README.md
│
├── 07-adaptive-rag/           # Dynamic retrieval
│   ├── app.py
│   └── README.md
│
├── 08-graph-rag/              # Graph-based retrieval
│   ├── app.py
│   └── README.md
│
├── 09-multihop-rag/           # Step-by-step retrieval reasoning
│   ├── app.py
│   └── README.md
│
├── 10-tool-augmented-rag/     # SQL/API + vector DB hybrid
│   ├── app.py
│   └── README.md
│
└── 11-experimental/           # New ideas (memory, clustering, generative fusion, etc.)
    ├── memory-rag/
    ├── clustered-rag/
    └── generative-fusion-rag/

```
