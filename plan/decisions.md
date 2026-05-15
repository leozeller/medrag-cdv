# Project Decisions — medrag-cdv

Living document. Captures the methodological and technical choices made
during the medrag-cdv sprint. Each entry: **Decision**, **Why**, and
**Tradeoff** (what we accept by going this way). Pending decisions are
listed at the end.

---

## Methodology

### M1. Scope: entity-only, no aspect classification

**Decision:** Implement only entity-aware query enrichment. Aspect
classification (the second tuple element in CDV's `<entity, aspect>`
formulation) is dropped from the proposal.

**Why:** A second classifier with its own training and evaluation does
not fit a 1-week sprint. The entity-aware variant captures the core CDV
idea (structured query enrichment with biomedical knowledge) and is the
focus of this work.

**Tradeoff:** We cannot claim to evaluate the *combination* of entity +
aspect. The poster makes this scope reduction explicit so the contribution
remains honest.

### M2. Three retrievers compared, plus an LLM-only control

**Decision:** The research question is a three-way **retriever**
comparison: BM25, Dense, Entity-Aware — all under a held-constant LLM
generator, all sharing the `retrieve(query, k) -> list[Passage]`
interface. Additionally, an **LLM-only baseline** (`--retriever none`,
no retrieval, no passages in the prompt) is run as a **control
condition** — not a fourth retriever, but a diagnostic for the eval.

**Why:**
- BM25 is the classical lexical baseline. Dense is the modern semantic
  baseline. Entity-Aware is the CDV-inspired variant on top of Dense.
  The triple lets us separate "is dense better than lexical" from
  "is entity enrichment helpful on top of dense".
- LLM-only as control answers a different question: "does retrieval
  contribute meaningfully on top of the LLM's pre-training knowledge?".
  The smoke run on dense already exposed cases (e.g. Lennox-Gastaut
  syndrome) where the LLM answered correctly *despite* retrieval failure
  — having an LLM-only ROUGE-L number lets us quantify how much of the
  ROUGE-L signal is attributable to retrieval vs. to the LLM itself.

**Tradeoff:**
- Four runs (3 retrievers + 1 control) instead of three. ~30 % more
  compute time. Acceptable given the diagnostic value.
- On the poster, the **main comparison plot must remain the three
  retrievers**; LLM-only is presented as a control number in a
  side-panel or discussion box, not as a fourth bar. Otherwise we
  conflate two research questions ("which retriever is best?" vs. "do
  we need retrieval at all?").

### M3. Retrieval corpus: Document-Level, Disorders-only subset

**Decision:** Index at the document level — each `<Document>` in MedQuAD
becomes one logical "document" (its `<Answer>` texts concatenated and
chunked ~300 tokens with 50-token overlap). Restrict to entries with
`semantic_group == "Disorders"`, yielding ~3,485 source documents.

**Why:** Answer-level indexing would be trivial — the gold passage *is*
the literal answer text, recall@5 would be ~95% for all methods, no
differentiation. Crawling the actual NIH source documents would take 1–2
days of setup we don't have. Doc-level preserves the CDV spirit (find
the right document, not the right sentence) while staying tractable.

**Tradeoff:**
- Less realistic than crawling full NIH sources (a deployed system would).
- Multiple chunks from one document can land in top-k, which complicates
  per-chunk evaluation; we sidestep this by doing recall at the doc level
  (see M4).

### M4. Gold-mapping: doc-level recall

**Decision:** For each question, the gold doc is `qid.split("-")[0]`.
Recall@k = whether the gold `doc_id` appears in the set of unique
`doc_id`s of the top-k retrieved chunks. MRR uses the rank of the first
chunk whose `doc_id` matches.

**Why:** Doc-level recall is interpretable and aligns with the doc-level
index choice (M3). Chunk-level recall would require manual labelling of
which chunk is "the right one" per question, which we don't have time for.

**Tradeoff:** A retriever can score Recall@5 ✓ by finding the right doc
but the wrong chunk (e.g. Symptoms instead of Treatment). We turn this
weakness into a *feature*: the LLM runs on the actual retrieved chunks,
so a wrong chunk produces a low ROUGE-L — chunk-granularity errors surface
downstream.

### M5. Evaluation metrics

**Decision:**
- **Recall@k** and **MRR** at the doc level (see M4)
- **ROUGE-L F1** between generated answer and gold answer
- **Abstention rate** — fraction of answers containing "I don't know"
  / "not enough information" phrases (proxy for calibration)
- **Passage overlap** — fraction of answer tokens that also appear in
  the retrieved passages (proxy for groundedness)
- All metrics also reported **stratified by `qtype`** for the qualitative
  analysis
- Full eval on a hand-sampled ~200–300 questions from the Disorders subset
  (exact size in D4)

**Why:** Recall@k and MRR are clean retrieval-only metrics. ROUGE-L is
the standard generation metric but has known limits (see M8).
Abstention rate and passage overlap are cheap, no-extra-LLM-call proxies
for the things ROUGE-L misses (calibration and grounding). Stratification
by `qtype` surfaces *where* entity-aware specifically helps. The sample
is needed because the LLM generation step dominates wall-clock time.

**Tradeoff:**
- Sampling 200–300 instead of the full 15,842 Disorders questions reduces
  statistical power — confidence intervals on the metric deltas will be
  wider.
- Abstention rate is a substring match — it catches the phrases the
  prompt encourages but misses paraphrased uncertainty.
- Passage overlap is token-set-based, not semantic — it rewards verbatim
  paraphrasing of passages and undercounts faithful but reworded answers.

### M6. No training

**Decision:** All encoders (embeddings, scispaCy NER + linker) and the
LLM are used frozen. No fine-tuning, no continued pretraining.

**Why:** Within scope for a 1-week sprint. CDV's contribution was the
training approach (hierarchical BLSTM + self-supervised multi-task
training on Wikipedia). We substitute that with frozen pre-trained
encoders + query-side entity enrichment — explicitly the simplification
this project takes.

**Tradeoff:** We cannot claim that domain-specific representation
*learning* helps — only that domain-specific *encoding* (with frozen
models) plus structured query enrichment helps. The poster frames the
contribution accordingly.

### M7. Empty-answer filter in parser

**Decision:** Skip QA pairs whose `<Answer>` is empty or whitespace-only
at parse time.

**Why:** MedQuAD's MPlusDrugs / A.D.A.M. / Herbs subsets have answers
removed by the original authors for MedlinePlus copyright reasons.
Keeping them would break gold-passage lookup (no passage to retrieve)
and ROUGE-L (nothing to compare against).

**Tradeoff:** We drop the QA pairs in those subsets even though their
question text, CUIs, and qtypes are valid. They could be used for
question-side studies (e.g. entity-linker quality) but not for our
retrieval-and-generation loop. The restriction is documented in README
under "Modifications".

### M8. ROUGE-L is incomplete — supplement with abstention + grounding proxies

**Decision:** ROUGE-L F1 alone is not enough to evaluate RAG quality.
We supplement it with two cheap, no-extra-LLM proxies — *abstention rate*
and *passage overlap* (formal definitions in M5) — and document the
remaining gap with manual annotation of the 20 qualitative cases.

**Why:** The dense smoke surfaced cases where the LLM-only baseline
generated factually wrong but lexically plausible answers (e.g. claiming
HMERF is a muscular dystrophy). ROUGE-L scored those near the
retrieval-augmented variants because the wrong answer still shares
medical vocabulary with the gold. Without supplementary signals, the
ROUGE-L delta would underestimate retrieval's contribution.

- Abstention rate distinguishes a well-calibrated "I don't know" from
  a confident hallucination — both score equally low on ROUGE-L.
- Passage overlap distinguishes an answer grounded in retrieved
  passages from one fabricated by the LLM — even when both produce
  lexically similar output.
- Qualitative annotation of the 20 hand-picked cases (Phase 5) records
  explicit faithful-vs-hallucinated labels.

**Tradeoff:**
- Neither proxy is a true faithfulness measure. An LLM-as-judge
  approach (e.g. RAGAS, TruLens) would be stronger but adds doubled
  eval-time, and the judge LLM has its own hallucination risk. Skipped
  for the sprint; flagged as future work in the poster.
- The abstention phrases are English-language and prompt-specific —
  rephrased uncertainty ("hard to say without more info") is missed.

---

## Technical

### T1. Python 3.11 + uv

**Decision:** Python 3.11 pinned via `.python-version`. uv (≥0.11) as the
package manager. Lockfile `uv.lock` committed.

**Why:** scispaCy 0.5.4 (latest stable) caps at Python 3.11. uv is
faster than pip+requirements or poetry and produces deterministic
lockfiles by default.

**Tradeoff:** Python 3.11 is one version behind the current bleeding
edge. When scispaCy lifts the cap, the project should upgrade — but
that's post-poster cleanup, not sprint work.

### T2. Embedding model: BGE primary, S-PubMedBERT as optional ablation

**Decision:** Primary dense encoder is `BAAI/bge-base-en-v1.5`.
`pritamdeka/S-PubMedBert-MS-MARCO` is held as an optional ablation if
time permits at the end of the sprint (see D2).

**Why:** Two reasons, in order:

1. **Strategic — make Entity-Aware's contribution legible.** A
   general-purpose encoder like BGE does not automatically resolve
   biomedical synonyms (e.g. "IgA nephropathy" ↔ "IgA Glomerulonephritis").
   That gives the Entity-Aware variant room to demonstrate its added value
   via explicit UMLS linking. A domain-pretrained encoder like
   S-PubMedBERT would absorb some of those synonyms internally, which
   would reduce the observable lift from Entity-Aware on top of it — and
   weaken the poster narrative.
2. **Empirical (with caveats).** BGE-base-en-v1.5 is top-tier on the
   MTEB IR leaderboard, including biomedical BEIR subsets (TREC-Covid,
   NFCorpus). However, we know of **no direct BGE-vs-S-PubMedBERT
   comparison on MedQuAD specifically**, so the empirical case is
   suggestive, not conclusive.

**Tradeoff:**
- BGE is not biomedical-tuned, so BGE-only numbers cannot isolate the
  contribution of *domain knowledge in the encoder*. We accept this and
  position S-PubMedBERT as the follow-up that closes that loop if time
  permits at the end of the sprint.
- A reviewer could argue: "why not start from a domain encoder?" — our
  honest answer is the strategic point above: we chose a baseline that
  makes Entity-Aware's contribution most visible.

### T3. Entity linker: scispaCy `en_core_sci_md` + UMLS linker

**Decision:** scispaCy 0.5.4 with the medium-size biomedical NER model
(`en_core_sci_md`) plus the built-in UMLS entity linker. Linker
confidence threshold pending (see D3).

**Why:** Open-source, no API key, runs locally. Standard choice in
biomedical NLP pipelines. The linker fetches a ~1 GB knowledge base on
first use, which is staged at setup (see T9), not at runtime.

**Tradeoff:** scispaCy's UMLS linker uses approximate nearest-neighbor
matching and can return noisy CUIs for short or ambiguous spans. We
mitigate via the confidence threshold (D3) and accept that some queries
will get no useful enrichment — these become a category in the qualitative
analysis.

### T4. LLM generation: Ollama with `medgemma1.5:4b`, held constant

**Decision:** Use Ollama (running on the user's host) for local LLM
hosting via `langchain-ollama`. Model: **`medgemma1.5:4b`** — Google's
medical-domain fine-tune of Gemma 3 (4.3B params, Q4_K_M quantization,
~3.3 GB). The model is held **constant** across all three retrievers.
GPU upgrade path to `medgemma:27b` remains open if user's cluster access
proves stable (see D1).

**Why:**
- Local hosting via Ollama avoids API costs, rate limits, and data
  egress concerns.
- Holding the LLM constant isolates retriever effects from generation
  effects — the experimental comparison is about retrievers, not LLMs.
- `medgemma1.5:4b` over a general 4B model (e.g. `gemma2:2b`) because:
  - Domain-fine-tuned → better answer quality on medical questions
  - Same compute footprint, CPU-realistic
  - Stronger poster story ("medical-domain LLM held constant")
- Crucially, unlike the embedding-encoder choice (T2), a domain-aware
  *generator* does not erode the Entity-Aware retriever's relative lift —
  generator domain affects absolute ROUGE-L, not the relative retriever
  ranking.

**Tradeoff:**
- License is the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms),
  not Apache 2.0. Research use only, not for clinical decision support.
  Documented in README under "Model attribution".
- Local LLM hallucination is higher than frontier API models. Absolute
  ROUGE-L scores will be lower than what e.g. GPT-4o would produce.
  Mitigated by reporting deltas between retrievers.

### T5. Vector store: FAISS `IndexFlatIP`

**Decision:** FAISS in-memory exact index, inner-product on L2-normalised
embeddings (equivalent to cosine similarity).

**Why:** ~15–25k chunks × 768 dim → ~50 MB index, queries are <10 ms.
No need for IVF / HNSW / PQ approximations at this scale.

**Tradeoff:** Won't scale to millions of vectors — but that's irrelevant
here. If the corpus were extended to full NIH crawls or PubMed, we'd
swap in an approximate index.

### T6. Chunking: ~300 tokens, 50-token overlap

**Decision:** Use LangChain's token-based recursive splitter with ~300
tokens per chunk and 50-token overlap.

**Why:** Standard RAG defaults. 300 tokens × top-5 ≈ 1,500 tokens of
context — fits comfortably in Ollama's default 4,096 context window.
Overlap of 50 keeps sentence boundaries from being cut hard.

**Tradeoff:** Larger chunks (500 tokens) give more context per hit but
waste budget on irrelevant text. Smaller chunks (150 tokens) give finer
recall granularity but more candidates to rank. 300 is a defensible middle
without a chunk-tuning experiment we don't have time for.

### T7. Retriever interface: `retrieve(query, k) -> list[Passage]`

**Decision:** All three retrievers expose the same callable signature
returning a list of `Passage` Pydantic models (with `doc_id`, `chunk_id`,
`text`, `score`).

**Why:** Identical signature → swappable in the pipeline without
`isinstance`-style branching. Pydantic carries IDs through so gold-mapping
(M4) is a pure data operation.

**Tradeoff:** A common signature means retriever-specific knobs (BM25's
`k1` and `b`, dense's similarity threshold) can't be exposed at call
time. We set them at construction instead and accept that they're not
per-query tunable — which is the right call for a comparison study anyway.

### T8. Prompt template: locked after Phase 2

**Decision:** Two locked prompt templates, both finalised on 2026-05-15:

1. **With passages** (BM25 / Dense / Entity-Aware):
   ```
   You are a medical question answering assistant. Answer the question
   based on the provided passages. Use only information from the passages
   — do not add facts from your own training. If the passages do not
   contain enough information, say so. Keep the answer concise and factual.
   ```
2. **Without passages** (LLM-only baseline):
   ```
   You are a medical question answering assistant. Answer the question
   concisely and factually from your own knowledge. If you don't know
   the answer with confidence, say "I don't know" rather than guess.
   ```

Implemented as `PROMPT` and `NO_CONTEXT_PROMPT` in `src/generation.py`.
No further changes for the rest of the sprint.

**Why:**
- Tuning the prompt would confound retrieval-quality effects with
  prompt-engineering effects.
- The "Use only information from the passages — do not add facts from
  your own training" instruction is intentionally stronger than the
  original draft, because the dense smoke showed the LLM ignoring
  passages and falling back on pre-training (e.g. Lennox-Gastaut).
- The two prompts share persona, tone, and concision instruction so
  that the LLM-only control and the retrieval variants are comparable.

**Tradeoff:**
- A locked prompt may not be optimal for the chosen LLM, so absolute
  ROUGE-L scores are below what a tuned prompt would yield. Acceptable
  because we report *deltas* between retrievers, not absolute quality.
- The "use only passages" instruction is enforced only by prompt — the
  model can still ignore it. Visible in qualitative review.

### T9. Artifacts pre-staged, never fetched at runtime

**Decision:** All third-party artifacts (MedQuAD data, embedding model,
scispaCy `en_core_sci_md` + UMLS linker, Ollama LLM weights) are
downloaded into their respective caches **before** running the pipeline.
Pipeline code reads from local cache paths only and sets offline-mode
env vars (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) where applicable.

**Why:** Reproducibility. A `uv run python -m src.pipeline` invocation
should never depend on network availability, transient hub outages, or
cache invalidation. Pre-staging also keeps the runtime free of long
first-download stalls.

**Tradeoff:** Setup is more explicit — the user must download all
artifacts once before the first pipeline run. We document the steps in
the README and accept this one-time cost for deterministic later runs.

### T10. Shared chunking module for all retrievers

**Decision:** Chunking logic lives in `src/chunking.py` (`prepare_chunks`),
used by `src/indexing.py` (FAISS dense index) and by
`src/retrievers/bm25.py` (in-memory BM25). All retrievers operate on the
*identical* chunk pool with identical metadata.

**Why:** Apples-to-apples comparison. If dense and BM25 chunked
differently, the recall deltas would conflate retriever quality with
chunking choice. By forcing a single chunking source, only the retrieval
mechanism varies. The entity-aware retriever reuses the dense index
directly (same chunks), so the constraint extends to all three.

**Tradeoff:** Each retriever loads the chunk pool independently — no
shared in-memory cache. At 24k chunks / ~26 MB input JSONL this is a
few seconds per startup, acceptable for sprint-scale eval runs.

### T11. BM25 index built in memory at runtime

**Decision:** Unlike the FAISS index (persisted to
`results/indices/dense/`), the BM25 index is rebuilt from the chunk
pool every time `--retriever bm25` runs. No disk persistence.

**Why:** 24k chunks × `BM25Okapi` index build takes ~5 seconds.
Persisting to disk and loading would save seconds at the cost of code
surface (serialization, cache invalidation when chunks change). Not
worth the complexity at this scale.

**Tradeoff:** If we scaled the corpus 10× (full MedQuAD or NIH crawl),
build time becomes noticeable and persistence should be added.

### T12. Pipeline output is resumable JSONL

**Decision:** Pipeline writes one JSON record per question to
`results/runs/{retriever}.jsonl` in **append mode**. On re-run, qids
already present in the output file are skipped (`--no-resume` overrides).

**Why:** Full eval runs take 30 min – 2 h on CPU with `medgemma1.5:4b`.
A network glitch, SSH-tunnel drop, OOM kill, or accidental Ctrl-C
shouldn't force restarting from question 1. With `fh.flush()` after
every completed record, the on-disk state always reflects committed
progress.

**Tradeoff:** If a retriever or generator change is *intentional* (e.g.
prompt change, different LLM), the user must `--no-resume` or delete
the file explicitly — otherwise stale results survive across runs.

### T13. MedGemma thinking-token stripping

**Decision:** Generated answers are post-processed to remove MedGemma
1.5's chain-of-thought trace. Paired `<unused94>…<unused95>` blocks
are regex-stripped; if only an opening tag is present, fall back to
keeping the last paragraph. See `src/generation.py::_strip_thinking`.

**Why:** MedGemma 1.5 emits an internal thinking trace before the
final answer. Including it in ROUGE-L would inflate scores artificially
(the trace echoes the question) and clutter qualitative review.

**Tradeoff:** The fallback ("last paragraph") could keep unwanted
content if the model formats thinking with extra paragraph breaks. We
can tune the regex if a sample of generated outputs shows leaks.

---

## Tunable Parameters (Stellschrauben)

These are settings that are **fixed now** but could legitimately be
re-tuned if the eval reveals problems. Tracked here so we know what knobs
exist and what each one costs to change.

### S1. LLM output cap (`num_predict=512`)

**Current:** 512 tokens. Hard cap on generated tokens per call,
≈ 50 sec wall-clock on CPU.

**Tradeoff:** ~14 % of gold answers in MedQuAD are longer than 512
tokens. But MedGemma:4b's typical generated answer is <100 tokens —
the cap is a safety net against runaway generations (one BM25 question
ran 10+ min before we added the cap) rather than a normal constraint.
Where it does bite, ROUGE-L is already low for those rows because the
LLM was answering concisely anyway.

**When to revisit:** if qualitative review shows a substantial fraction
of generated answers ending exactly at 512 tokens. Bump to 1024 (still
hard-capped, but covers p95 of gold answers).

### S2. Retrieval top-k (`-k 3`)

**Current:** 3 passages per query.

**Tradeoff:** Reduced from 5 to keep total input prompt under
`num_ctx=2048` (3 × ~360 tokens + system + question + generation
budget ≈ 1500 tokens). Fewer passages means: if the right content is
in passage 4 or 5, we miss it. Recall@5 on retrieval is shown as Recall@3
effectively (the eval CLI uses `DEFAULT_K=3` to match).

**When to revisit:** if recall numbers are systematically low and a
qualitative spot-check shows the "right" passage was at rank 4 or 5.
Then either bump `-k` to 5 and raise `num_ctx` to 4096 (accepting the
inference-speed cost), or chunk smaller so 5 passages still fit.

### S3. Ollama `num_thread=8`

**Current:** 8 threads per request, passed by the `ChatOllama` client.

**Tradeoff:** With 2 threads the LLM ran at 200 % CPU and a single
inference took 2–4 min. With 8 threads it's at ~800 % CPU and
~30 sec/question. Beyond 8 threads, llama.cpp inference becomes
memory-bandwidth-bound — more threads don't help and sometimes hurt
because of cache contention.

**When to revisit:** if the user's machine has noticeably more or fewer
cores. 8 is a reasonable default for any modern multi-core CPU.

### S4. Ollama `num_ctx=2048`

**Current:** 2048 tokens of context window.

**Tradeoff:** Matches Ollama's own default — using anything other than
the default triggers a **model reload** on the first request, which
costs minutes of cold-start latency on CPU. With our `-k 3` setting,
input prompts stay ~1500 tokens which fits comfortably.

**When to revisit:** if we want to increase `-k` (see S2) or pass much
longer passages, raise to 4096. Pay the one-time reload cost; subsequent
requests run at the new context size.

### S5. Ollama `keep_alive="1h"`

**Current:** 1 hour. Overridable via `OLLAMA_KEEP_ALIVE` env var.

**Tradeoff:** Ollama's own default is 5 min — too short for our
multi-retriever workflow where there's often a >5 min gap between
runs (smoke check, eval, decide next step). 1h covers the typical
interactive session. Set `OLLAMA_KEEP_ALIVE=-1` for unattended
overnight full-eval runs so the model never unloads.

**When to revisit:** if RAM pressure ever matters (~3.3 GB held for
medgemma:4b). Reduce to `5m` or stop unsetting.

---

## Pending Decisions

### D1. Ollama model — ✅ Resolved (2026-05-14)

**Resolution:** `medgemma1.5:4b` on CPU. See **T4** for full rationale
and license notes. GPU upgrade path to `medgemma:27b` open conditional
on stable cluster access.

### D2. S-PubMedBERT ablation

**Status:** Conditional. Run as a second eval pass if Mon/Tue have
spare time after the qualitative analysis is done. Otherwise omit and
mention as future work on the poster.

### D3. scispaCy linker confidence threshold — ✅ Resolved (2026-05-14)

**Resolution:** Default `0.85`, tunable via the `--entity-threshold`
CLI flag on `src.pipeline`. May still revisit if a sample of MedQuAD
entity extractions shows noisy linkages.

### D4. Full-eval sample size — ✅ Resolved (2026-05-14)

**Resolution:** 300 questions, sampled deterministically from the
Disorders subset with `random.Random(seed=42)`. Use `--limit N` to
shrink for smoke tests; `--sample-size N` to override the test set
size entirely.

### D5. Streamlit demo

**Status:** Out by default. Reintroduce only if everything else is done
ahead of schedule.
