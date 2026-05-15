# Team Project Plan — Entity-Aware Medical RAG

Internal working document. Not for the lecturer.

## Status (2026-05-15)

- Phase 0 ✅ Setup + uv scaffold
- Phase 1 ✅ MedQuAD parser, 16k records
- Phase 2 ✅ Dense retrieval + LLM generation pipeline
- Phase 3 ✅ BM25 retriever + entity-aware retriever
- Phase 4 ✅ Evaluation: Recall@k, MRR, ROUGE-L, abstention rate,
  passage overlap (`2db97fe`)
- ✅ Generation tuning + LLM-only control baseline (`ffdff47`):
  switched to `medgemma:4b` (no thinking, ~3× faster), hardened
  with-passages prompt, runtime caps (`num_predict`, `num_thread`,
  `keep_alive`)
- ✅ FAISS index built; all four smoke runs pass with finalised prompt;
  entity-aware shows R@3=1.0 / MRR=0.867 on the 5-question smoke
- 🔄 Full eval started: 300 questions × 4 runs (dense, bm25, entity,
  none), ~8h total
- Next: qualitative analysis + poster

## Status (2026-05-14)

Poster deadline: 2026-05-21.

- Phase 0 ✅ Setup + uv scaffold (`6f421b5`)
- Phase 1 ✅ MedQuAD parser, 16k records (`761fe01`)
- Phase 2 ✅ Dense retrieval + LLM generation pipeline (`64b65ca`)
- Phase 3 ✅ BM25 (`1dfc810`) + entity-aware skeleton (`17e3d98`)
- Phase 4 ✅ Evaluation module: Recall@5, MRR, ROUGE-L (`dd7a270`)
- Next: full pipeline runs (3 retrievers × ~300 questions) once the
  FAISS index is built and scispaCy `en_core_sci_md` is installed
- Then: qualitative analysis + poster

---

## Roles

- **Person A:** retrieval pipeline (indexing, three retrievers, query enrichment)
- **Person B:** generation, evaluation, qualitative analysis

Pair-program in week 1 days 1–2 (architecture). Then split. Cross-review at end of week 1 and week 2.

## Tech stack

Already known: LangChain, Ollama
New: FAISS, scispaCy + UMLS linker, sentence-transformers (via LangChain `HuggingFaceEmbeddings`)

## Repository layout (decide together day 1)

```
medrag-cdv/
├── data/
│   ├── medquad/             # raw download
│   └── processed/           # parsed JSON: {question, answer, entity_cui, aspect, doc_id}
├── src/
│   ├── indexing.py          # build FAISS + BM25 indices
│   ├── retrievers.py        # BM25, Dense, EntityAware — same interface
│   ├── generation.py        # Ollama call + fixed prompt template
│   ├── pipeline.py          # ties retriever + generator together
│   └── evaluation.py        # Recall@k, MRR, ROUGE-L
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pipeline_smoke_test.ipynb
│   └── 03_qualitative_analysis.ipynb
├── results/
│   ├── runs/                # per-retriever output JSON
│   ├── metrics.csv
│   └── qualitative.md
├── README.md
└── pyproject.toml
```

---

## Week 1 — Pipeline skeleton

### Mon–Tue (pair programming, both)
- [ ] Repo + environment setup
- [ ] Download MedQuAD from `github.com/abachaa/MedQuAD`
- [ ] Parse XML → JSON, save to `data/processed/`
- [ ] `ollama pull biomistral` (or `llama3.1:8b`) — test with one question
- [ ] **Decide and commit the prompt template** (do not change later)
- [ ] Agree on the retriever interface: `retrieve(query: str, k: int) -> List[Passage]`

### Wed–Thu (Person A leads)
- [ ] Document chunking (~300 tokens, 50 overlap)
- [ ] FAISS index with `HuggingFaceEmbeddings` (S-PubMedBERT or BGE)
- [ ] Dense retriever wrapped to fit the agreed interface
- [ ] LangChain RetrievalQA chain working end-to-end on 10 sample questions

### Wed–Thu (Person B parallel)
- [ ] Evaluation skeleton: function `run_eval(retriever, test_set) -> outputs.json`
- [ ] Recall@k and MRR functions (`ranx` library or implement manually)
- [ ] ROUGE-L via `evaluate` library, smoke-tested on 5 examples

### Fri (Person A)
- [ ] BM25 retriever via LangChain `BM25Retriever`, same interface
- [ ] Both retrievers swappable in the pipeline

### Fri (Person B)
- [ ] Run evaluation on 50 questions for both retrievers as smoke test
- [ ] Verify outputs.json format works for the metric scripts

**End-of-week goal:** BM25 and dense both produce answers + saved outputs for sample questions. Pipeline is end-to-end.

---

## Week 2 — Entity-aware variant + full evaluation

### Mon–Tue (Person A leads)
- [ ] Install scispaCy + `en_core_sci_md` + UMLS linker
- [ ] Function: `extract_entities(question: str) -> List[Entity]`
- [ ] Query enrichment: append entity to query string
- [ ] EntityAware retriever — third class, same interface

### Mon–Tue (Person B parallel)
- [ ] Plotting script: bar chart for results table
- [ ] Qualitative analysis template (markdown, one block per case)

### Wed (joint)
- [ ] Run all three retrievers on the full ~300-question set
- [ ] Save outputs to `results/runs/{bm25,dense,entity}.json`

### Thu (Person B leads)
- [ ] Compute Recall@5, MRR, ROUGE-L for all three
- [ ] Generate `metrics.csv` and the bar plot

### Fri (joint)
- [ ] Pick 20 representative questions for qualitative analysis
- [ ] Each writes commentary on 10 cases
- [ ] Discuss and finalize together

**End-of-week goal:** All numbers final. No more code changes. Results frozen.

---

## Week 3 — Poster + presentation

### Mon–Tue
- [ ] Pipeline diagram (excalidraw, draw.io, or LaTeX-tikz)
- [ ] Choose poster template (LaTeX `betterposter` or PowerPoint A0 template)
- [ ] First layout draft

### Wed–Thu
- [ ] Fill in: results table, three qualitative example boxes, conclusion
- [ ] Iterate on layout (someone external should look at it)
- [ ] Practice 2-minute elevator pitch (each)

### Fri
- [ ] Poster finalized, sent to print **at least 24h before session**
- [ ] **Optional:** Streamlit demo if time allows (one half-day max)

**Poster session: end of week.**

---

## Week 4 — Polish

- [ ] Address feedback from poster session
- [ ] README: setup instructions, how to reproduce results, what each script does
- [ ] Code cleanup: type hints, docstrings on main functions
- [ ] Final results document (extend the proposal with actual numbers + discussion)

---

## Watch-outs (read once, look back later)

**UMLS linker setup is annoying.** scispaCy downloads the UMLS knowledge base on first use (~1 GB). Do this on day 1, not day 8. The `nmslib` dependency sometimes needs special install flags on M1/M2 Macs — check early.

**Ollama context window.** Default is often 2048 tokens. Top-5 medical passages can blow this. Set `num_ctx=4096` (or 8192) explicitly in the Ollama call.

**Determinism.** Set `temperature=0` for Ollama. Save all generation outputs to disk. Never re-run when you only need to recompute metrics.

**Don't tune the prompt.** Pick one template in week 1, day 2. Never touch it again. Otherwise you cannot fairly compare retrievers — that's the whole point.

**Two-person integration risk.** Run each other's code on each other's machine at the end of each week. Don't trust "works on my machine."

**Streamlit is optional.** If week 3 is tight, drop the demo before dropping any results work.

---

## Definition of done — implementation

- [ ] Three retrievers, same interface, swappable in the pipeline
- [ ] Same generator, same prompt across all three
- [ ] Results table: three rows × three metric columns
- [ ] 20 qualitative cases written up
- [ ] Everything reproducible from `python -m src.pipeline --retriever {bm25|dense|entity}`
