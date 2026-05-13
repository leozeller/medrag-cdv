# Project Proposal: Entity-Aware Medical RAG

**Team:** Leonie Zeller, Naeem Bashir

---

## Idea

We want to build a medical question-answering system based on Retrieval-Augmented Generation (RAG), inspired by Arnold et al. (WWW 2020), *Learning Contextualized Document Representations for Healthcare Answer Retrieval*. The original paper proposes representing medical queries as structured `<entity, aspect>` tuples — for example, `<IgA nephropathy, symptoms>` — rather than as plain natural-language strings. Their model retrieves passages whose contextualised sentence representations match both components of the tuple simultaneously.

We will adopt the entity-aware part of this idea in a simplified form built on modern pre-trained encoders, replacing the original hierarchical BLSTM and self-supervised multi-task training with a query-side enrichment strategy that uses frozen domain models throughout.

## CDV-Inspired Component

For each incoming question, our pipeline performs two steps before retrieval:

**1. Entity extraction.** scispaCy with the UMLS entity linker detects medical entities in the question and resolves them to their canonical UMLS Concept Unique Identifiers (CUIs). For example, the surface form *"IgA nephropathy"* resolves to the entity `C0017661` with canonical name *"IgA Glomerulonephritis"*. Confidence scores from the linker are used to filter false positives.

**2. Query enrichment.** The original natural-language question is augmented with the resolved entity into a structured query string, e.g. `[Entity: IgA Glomerulonephritis (C0017661)] What are the early signs?`. The pre-trained sentence encoder, trained on biomedical text, picks up this explicit signal at retrieval time.

This is the simplification of CDV's structured query model: instead of training a custom encoder to learn the tuple representation, the entity is made explicit in the query string and a strong frozen encoder handles the matching. The trade-off is that we focus on the entity dimension and omit the long-range document discourse modelling of the original BLSTM; the gain is substantial reduction in implementation complexity, allowing all components to be built on top of pre-trained, off-the-shelf models.

**Note on aspect modelling.** A natural extension would be to additionally classify each question into one of approximately ten clinical aspects (symptoms, treatment, diagnosis, causes, prognosis, prevention, risk factors, complications, epidemiology) corresponding to the most frequent section headings in medical Wikipedia identified by Arnold et al. (Table 1), and to append the aspect label to the enriched query alongside the entity. This would yield the full `<entity, aspect>` representation of the original CDV approach. We have considered this extension but leave it out due to time constraints: the additional classification component, its evaluation, and its integration are not feasible within the 4-week timeframe. The entity-aware variant captures the core CDV idea and is the focus of this proposal.

## Approach

- **Data:** MedQuAD (~300 disease questions, ~150 NIH source documents, with UMLS entity annotations)
- **Retrievers (compared):** BM25 / dense (S-PubMedBERT) / entity-aware
- **Generation:** locally-hosted Gemma2-2b via Ollama, held constant across all retriever variants
- **Stack:** LangChain, FAISS, scispaCy, Ollama

## Evaluation

- **Retrieval:** Recall@5 and MRR against MedQuAD gold passages
- **Answer quality:** ROUGE-L F1 against gold answers
- **Qualitative analysis:** 20 representative cases inspected by hand, with focus on cases where the entity-aware variant helps, hurts, or makes no difference

## References

- Arnold, S., van Aken, B., Grundmann, P., Gers, F. A., Löser, A. (2020). *Learning Contextualized Document Representations for Healthcare Answer Retrieval.* WWW '20.
- Ben Abacha, A., Demner-Fushman, D. (2019). *A Question-Entailment Approach to Question Answering.* BMC Bioinformatics 20(1).
- Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Neumann, M., King, D., Beltagy, I., Ammar, W. (2019). *ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing.* BioNLP Workshop.
