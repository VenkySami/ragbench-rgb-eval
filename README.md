# ragbench-rgb-eval

**AIML PG-Level Capstone Project — Group 8**
Retrieval-Augmented Generation: Robustness & Reliability Across the RAGBench
and RGB Benchmarks.

This repository is a two-project monorepo covering both benchmark tracks
used to evaluate our RAG pipeline:

```
ragbench-rgb-eval/
  ragbench/   RAGBench track  — domain-wise pipeline benchmarking (chunking,
              embedding, FAISS indexing) across 5 real-world domains, scored
              end-to-end with the TRACe framework (Relevance, Utilization,
              Adherence, Completeness).
  rgb/        RGB track       — diagnostic robustness evaluation (Noise
              Robustness, Negative Rejection, Information Integration,
              Counterfactual Robustness) with baseline vs. mitigated
              comparisons across Llama, Mixtral, and Gemma via Groq.
```

Each track is self-contained with its own README, dependencies, and run
instructions:

- [`ragbench/README.md`](ragbench/README.md) — RAGBench data-preparation
  pipeline (Colab notebooks: dataset prep → chunking → embedding → FAISS
  indexing).
- [`rgb/README.md`](rgb/README.md) — RGB diagnostic evaluation framework
  (Python package: experiment runners, mitigations, metrics, CLI).

## Why two tracks?

A single "overall accuracy" number hides more than it reveals about a RAG
system's reliability. RAGBench validates **holistic, end-to-end answer
quality** across real-world domains, while RGB **diagnostically stress-tests**
specific robustness abilities (noise tolerance, calibrated abstention,
multi-document synthesis, counterfactual resistance) that a single accuracy
metric cannot isolate. Running both against comparable pipeline components
gives a much more complete picture of system reliability than either
benchmark could provide alone — see the capstone's final report and
presentation for the full cross-track analysis.

## Repository layout & lightweight policy

Both tracks generate large intermediate artifacts (embeddings, FAISS
indexes, LLM response caches, result CSVs). To keep this repository fast to
clone, only **source code, notebooks, configuration, and small summary
manifests** are committed — bulky generated artifacts are excluded via
[`.gitignore`](.gitignore) and are reproducible by re-running the documented
pipelines/scripts in each track's README.

## License

See [LICENSE](LICENSE).
