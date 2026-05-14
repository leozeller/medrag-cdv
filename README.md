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

# Pull the local LLM
ollama pull medgemma1.5:4b
```

## Data

MedQuAD ([github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD))
is **not committed** to this repo. Clone it into `data/medquad/`:

```bash
cd data && git clone https://github.com/abachaa/MedQuAD.git medquad
```

Then parse the XML corpus to a JSONL workfile:

```bash
uv run python -m src.data.parse_medquad
```

The parser walks `data/medquad/*/*.xml`, normalises whitespace, drops
QA pairs with empty `<Answer>` fields, and writes one JSON record per
QA pair to `data/processed/medquad.jsonl` (~16k records, ~26 MB).

## Data attribution

MedQuAD is published under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
If you use this code or its outputs, cite the original paper:

> Asma Ben Abacha and Dina Demner-Fushman.
> *A Question-Entailment Approach to Question Answering.*
> BMC Bioinformatics, 20(1):511, 2019.
> <https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4>

**Modifications.** We use the data in transformed form: XML records are
parsed to JSONL, whitespace is normalised, and QA pairs with empty
`<Answer>` fields are dropped.

## Model attribution

**LLM generator: `medgemma1.5:4b`** (Google MedGemma 1.5, 4B parameters).
A Gemma 3 derivative fine-tuned by Google on medical text. Pulled locally
via Ollama; held constant across all three retrievers.

- Model card: <https://huggingface.co/google/medgemma-4b-it>
- License: [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).
  Research use only — **not for clinical decision support or any
  patient-facing deployment**.

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
