# RGB Capstone — RAG System Evaluation on the RGB Benchmark

Implements the task in `Tasks/RGB Task-RAG Capstone Project.pdf`:
evaluate **Noise Robustness**, **Negative Rejection**, **Information Integration**,
and **Counterfactual Robustness** (Chen et al., 2023 — RGB, arXiv:2309.01431)
on 3 free-tier LLMs, then apply and measure lightweight mitigations for the
limitations the paper itself identifies for each ability.

This is an independent re-implementation (own prompt wiring, own sampling code,
own metric formulas). The [reference repo](https://github.com/chen700564/RGB) was
only read for the data schema — no code was copied from it.

## 1. Setup

```bash
cd ragbench-rgb-eval/rgb
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the 3 free API keys below
```

### Free-tier API keys (no OpenAI/ChatGPT/ChatGLM — those are the models the
task PDF/reference repo already used)

| Model (env var)                         | Provider          | Free tier                         |
|------------------------------------------|-------------------|------------------------------------|
| `GEMINI_API_KEY` → gemini-2.0-flash       | Google AI Studio  | https://aistudio.google.com/apikey |
| `GROQ_API_KEY` → llama-3.3-70b-versatile  | Groq              | https://console.groq.com/keys      |
| `MISTRAL_API_KEY` → mistral-small-latest  | Mistral La Plateforme | https://console.mistral.ai/api-keys |

All three have generous no-credit-card free tiers as of 2025.

Data files (`en_refine.json`, `en_int.json`, `en_fact.json`) are already in
`data/raw/` (pulled from the RGB repo's `master` branch). Re-download with:
```bash
python scripts/download_data.py
```

## 2. Project layout

```
src/rgb_eval/
  config.py            # constants: passage_num=5, noise rates, models, phrases
  data_loader.py        # parses en_refine/en_int/en_fact.json + noise-rate sampling
  prompts.py             # exact Figure 3 system/user prompt templates
  llm_clients.py         # Gemini / Groq / Mistral clients, unified interface + disk cache
  matching.py             # answer/refusal/factual-error matching helpers (hand-written)
  metrics.py               # accuracy / rejection rate / ED / CR formulas (hand-written)
  experiments/            # one runner per ability (baseline, Figure 3 prompt only)
  mitigations/            # one improvement technique per ability (see section 4)
scripts/
  download_data.py
  run_experiment.py       # CLI: pick ability x model x mitigation on/off
  aggregate_results.py    # builds the paper-style summary tables from results/*.csv
tests/                    # unit tests for metrics/matching/sampling (no API calls)
```

## 3. Running

```bash
# Baseline, single ability/model:
python scripts/run_experiment.py --ability noise_robustness --model gemini --sample-size 60
python scripts/run_experiment.py --ability negative_rejection --model groq --sample-size 60
python scripts/run_experiment.py --ability information_integration --model mistral --sample-size 60
python scripts/run_experiment.py --ability counterfactual_robustness --model gemini --sample-size 60

# All 3 models x all 4 abilities (baseline):
python scripts/run_experiment.py --ability all --model all --sample-size 60

# With mitigation applied (see section 4), to measure improvement:
python scripts/run_experiment.py --ability all --model all --sample-size 60 --mitigation

# Build paper-style summary tables from everything in results/:
python scripts/aggregate_results.py
```

Every LLM call is cached on disk (`cache/`) keyed by (model, ability, mitigation flag,
prompt) — reruns and `aggregate_results.py` are free and idempotent. Start with a
small `--sample-size` (e.g. 20-30) to validate wiring cheaply, then scale up
(e.g. 100-300) for stable numbers — safe to re-run because of caching.

## 4. Limitations identified in the paper, and the mitigation implemented for each

The RGB paper's own error analysis / discussion sections point to concrete causes.
We picked one lightweight, explainable mitigation per ability and measure the
before/after delta as part of the results (`--mitigation` flag):

| Ability | Paper's diagnosis of the failure | Mitigation implemented | File |
|---|---|---|---|
| Noise Robustness | Errors from long-distance information, evidence uncertainty, concept confusion — the model can't tell relevant snippets from noisy ones once mixed together | **Relevance-filter-before-answer**: a cheap lexical-overlap pre-filter scores each retrieved doc against the query and drops the bottom-scoring docs before they ever reach the answering prompt, shrinking the noise the model has to reason over | `mitigations/noise_filter.py` |
| Negative Rejection | LLMs "fail to strictly follow instructions" and produce unpredictable text instead of a clean refusal when no doc supports the answer | **Explicit verification-then-answer prompt**: a second, stricter instruction forces the model to first state yes/no whether *any* document answers the question before generating a final response, tightening the refusal signal | `mitigations/decision_calibration.py` |
| Information Integration | Merging/Ignoring/Misalignment errors — the model can't decompose a multi-part question; paper explicitly suggests chain-of-thought decomposition (noting it costs latency) | **Query decomposition**: split the compound question into sub-questions, answer each against its own docs, then merge into a final answer — trading extra calls for accuracy | `mitigations/decompose_integrate.py` |
| Counterfactual Robustness | Paper: "existing LLMs do not have a safeguard... they heavily depend on the information they retrieve" even when their own parametric knowledge is right | **Parametric-vs-retrieved cross-check**: ask the model its own no-context answer first, then separately ask it to compare that against the retrieved documents and explicitly flag/resolve any conflict, instead of answering from the docs alone | `mitigations/fact_check.py` |

Section 5 of the presentation reports baseline vs mitigated numbers for all four
abilities across the 3 models, so "how to make it better" is answered with data,
not just narrative.

## 5. Metrics (hand-written in `metrics.py`, not reused from the reference repo)

- **Accuracy** (Noise Robustness, Information Integration): fraction of examples
  where the response contains at least one gold answer alias.
- **Rejection Rate** (Negative Rejection, noise_rate=1.0): fraction of responses
  matching the refusal phrase (`Rej`, exact-phrase matching only — `Rej*` LLM-judged
  variant is explicitly excluded per task instructions).
- **Error Detection Rate (ED)**: fraction of counterfactual-document responses that
  contain the "There are factual errors..." phrase (paper's exact-match ED, not the
  LLM-judged ED*, which is also excluded per task instructions).
- **Error Correction Rate (CR)**: of the detected cases, fraction where the response
  also contains the correct (non-fake) answer.

## 6. Status

Core pipeline (data loading, prompts, LLM clients, matching, metrics, all four
experiment runners, and all four mitigations) is implemented and runnable end-to-end
via `scripts/run_experiment.py` and `scripts/aggregate_results.py`.
