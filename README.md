# medrag-cdv

Entity-aware medical Retrieval-Augmented Generation, inspired by
Arnold et al., *Learning Contextualized Document Representations for
Healthcare Answer Retrieval* (WWW 2020).

A small RAG pipeline over MedQuAD that compares three retrievers —
**BM25**, a **dense** baseline, and an **entity-aware** variant that
enriches the query with UMLS entities extracted by scispaCy — under
a held-constant local LLM generator.

## Setup

Requirements: [`uv`](https://docs.astral.sh/uv/) and a working
[Ollama](https://ollama.com/) installation on your machine.

```bash
# Install Python 3.11 + project dependencies into .venv/
uv sync

# Install the scispaCy biomedical model + UMLS linker
uv run pip install \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

# Pull the local LLM (default: gemma2:2b — small enough for CPU)
ollama pull gemma2:2b
```

## Data

MedQuAD is downloaded from <https://github.com/abachaa/MedQuAD>:

```bash
uv run python -m src.data.download_medquad
```

The raw XML is parsed into `data/processed/medquad.jsonl` (one
question/answer record per line, with UMLS CUI annotations).

## Running the pipeline

```bash
# Build the FAISS index
uv run python -m src.indexing

# Run a retriever end-to-end on the test subset
uv run python -m src.pipeline --retriever bm25
uv run python -m src.pipeline --retriever dense
uv run python -m src.pipeline --retriever entity
```

Results land in `results/runs/{bm25,dense,entity}.json`.

## Evaluation

```bash
uv run python -m src.evaluation \
    results/runs/bm25.json \
    results/runs/dense.json \
    results/runs/entity.json
```

Reports Recall@5, MRR, and ROUGE-L F1 for each retriever.

## Project layout

```
src/
├── indexing.py      # Chunking + FAISS index build
├── retrievers.py    # BM25, dense, entity-aware retrievers
├── generation.py    # Ollama call, fixed prompt template
├── pipeline.py      # CLI: retriever -> generator -> outputs
└── evaluation.py    # Recall@k, MRR, ROUGE-L
data/                # Raw + processed MedQuAD (gitignored)
results/runs/        # Per-retriever outputs
notebooks/           # Exploration + qualitative analysis
tests/
plan/                # Project planning docs
```

## References

- Arnold et al. (2020), *Learning Contextualized Document Representations
  for Healthcare Answer Retrieval.* WWW '20.
- Ben Abacha & Demner-Fushman (2019), *A Question-Entailment Approach
  to Question Answering.* BMC Bioinformatics 20(1).
- Neumann et al. (2019), *ScispaCy: Fast and Robust Models for
  Biomedical Natural Language Processing.* BioNLP Workshop.
