# RGB Capstone — Deep Technical Walkthrough

Audience: teammates who need to understand the codebase well enough to run it,
extend it, or defend it in review. This is a code-level companion to
`README.md` (which covers setup/usage) and `presentation/RGB_Capstone_Presentation.pptx`
(which covers results/story). This doc explains **how the system is built and why**.

---

## 1. What problem this repo solves

The RGB benchmark (arXiv:2309.01431) asks one question: *when you bolt
retrieval onto an LLM, does the model actually use the retrieved documents
well, or does it get confused, hallucinate, or blindly trust bad context?*
It decomposes "using retrieved context well" into 4 independently testable
abilities:

| Ability | Question it answers | Source file(s) | Metric |
|---|---|---|---|
| Noise Robustness | Can the model find the right answer when relevant docs are mixed with irrelevant ("noise") docs? | `en_refine.json` | Accuracy |
| Negative Rejection | Does the model correctly refuse to answer when **no** retrieved doc actually contains the answer? | `en_refine.json` (noise_rate=1.0) | Rejection Rate |
| Information Integration | Can the model combine facts spread across **multiple** documents to answer a compound question? | `en_int.json` | Accuracy |
| Counterfactual Robustness | Can the model detect a retrieved doc that's factually wrong and still give the correct answer? | `en_fact.json` | Error Detection Rate (ED), Error Correction Rate (CR) |

Everything in `src/rgb_eval/` exists to run these 4 experiments, twice each
(baseline vs. a mitigation), across 3 different LLMs, in a way that's cheap to
re-run and easy to audit.

---

## 2. Repo layout (what lives where)

```
RGB_Capstone/
├── data/raw/                     # en_refine.json, en_int.json, en_fact.json (from upstream RGB repo, data only)
├── cache/                        # disk cache of every LLM call, keyed by sha256(model|temp|system|user)
├── results/                      # CSV output of scripts/run_experiment.py + aggregate_results.py
├── presentation/                 # slide deck + chart assets
├── scripts/
│   ├── download_data.py          # pulls the 3 English JSON files from upstream RGB repo (data schema only)
│   ├── run_experiment.py         # CLI entrypoint: --ability x --model y --sample-size n [--mitigation]
│   └── aggregate_results.py      # merges results/*.csv into 4 paper-style pivot tables
├── src/rgb_eval/
│   ├── config.py                 # all constants: paths, noise rates, model registry, exact phrases
│   ├── data_loader.py            # JSON parsing + noise-controlled document sampling
│   ├── prompts.py                # Figure-3 prompt (verbatim) + all 4 mitigation prompt variants
│   ├── llm_clients.py            # unified client interface over Gemini / Groq / Mistral + disk cache + retry/throttle
│   ├── matching.py               # hand-written text-matching (answer alias / refusal / factual-error detection)
│   ├── metrics.py                # hand-written metric formulas (accuracy, rejection rate, ED, CR)
│   ├── experiments/               # one file per ability — the orchestration/glue layer
│   │   ├── noise_robustness.py
│   │   ├── negative_rejection.py
│   │   ├── information_integration.py
│   │   └── counterfactual_robustness.py
│   └── mitigations/               # one file per ability — the "improvement technique" layer
│       ├── noise_filter.py            (Noise Robustness)
│       ├── decision_calibration.py    (Negative Rejection)
│       ├── decompose_integrate.py     (Information Integration)
│       └── fact_check.py              (Counterfactual Robustness)
└── tests/                        # test package scaffold (currently only __init__.py — see §8 "known gaps")
```

**Design principle**: each experiment file (`experiments/*.py`) is a thin
orchestrator — it loads data, builds a prompt, calls a client, and scores the
response. All the "smart" logic lives in reusable, independently-testable
modules (`data_loader`, `prompts`, `matching`, `metrics`, `mitigations/*`).
This is what makes it possible to toggle `--mitigation` on/off without
duplicating the experiment loop.

---

## 3. End-to-end data flow

```
en_refine.json / en_int.json / en_fact.json
        │  (scripts/download_data.py — one-time)
        ▼
data_loader.load_records(ability)            # parse JSON → list[Record]
        │
data_loader.sample_records(records, N)        # deterministic seeded subsample
        │
for each noise_rate:
    data_loader.sample_docs(record, noise_rate, passage_num=5)
        │   deterministic mix of `positive` (relevant) + `negative` (noise) docs
        ▼
[optional] mitigations/noise_filter.filter_relevant_docs(query, docs)
        ▼
prompts.build_prompt(query, docs)             # → (system_instruction, user_prompt) — Figure 3, verbatim
        ▼
llm_clients.<Client>.generate(system, user)   # disk-cache check → API call (retry+throttle) → cache write
        ▼
matching.contains_answer / is_refusal / flags_factual_error   # score the raw text response
        ▼
metrics.accuracy / rejection_rate / error_detection_rate / error_correction_rate
        ▼
experiments/<ability>.run() returns list[dict] rows
        ▼
scripts/run_experiment.py → pandas.DataFrame → results/results_<ability>_<model>_<baseline|mitigated>.csv
        ▼
scripts/aggregate_results.py → results/summary_<ability>.csv (pivot tables, paper-style)
```

Every arrow above is a real function boundary — no hidden global state, no
stateful class beyond the LLM client (which only holds its own model id and
throttle timestamp).

---

## 4. Module-by-module deep dive

### 4.1 `config.py` — single source of truth for constants
Holds: `DATA_FILES` (ability → JSON path), `PASSAGE_NUM=5` (paper default:
5 docs shown per question), `NOISE_RATES=[0,0.2,0.4,0.6,0.8]` (Table 1 of the
paper), `INTEGRATION_NOISE_RATES=[0,0.2,0.4]` (Table 5), the **exact** refusal
and factual-error phrases the model is instructed to emit, and `MODELS` — the
logical name → concrete API model id mapping (`gemini` → `gemini-flash-latest`,
`groq` → `llama-3.1-8b-instant`, `mistral` → `mistral-small-latest`). Nothing
else in the codebase hardcodes these values — change a noise rate or a model
id in exactly one place.

### 4.2 `data_loader.py` — noise-controlled sampling (the trickiest logic)
`en_refine.json`/`en_int.json` records carry two document pools per question:
`positive` (docs that actually contain the answer) and `negative` (irrelevant
"noise" docs, same topic but no answer). `en_int.json`'s `positive` is a
**list of lists** — one list of snippets per sub-question of a multi-hop
question — so `_flatten()` normalizes that shape before sampling.

`sample_docs(record, noise_rate, passage_num)` is the noise-injection engine:
```python
num_positive = round(passage_num * (1 - noise_rate))
num_negative = passage_num - num_positive
```
e.g. at `noise_rate=0.4`, `passage_num=5` → 3 positive + 2 negative docs are
sampled from their respective pools, shuffled together, and returned as the
5-document context. Sampling uses `random.Random(f"{record.id}-{noise_rate}-{SEED}")`
— a **per-(record, noise_rate) deterministic seed** — so the exact same
document set is regenerated on every re-run. This is what makes disk caching
correct: identical (model, prompt) pairs really do recur across runs, so the
cache hit rate is high and results are exactly reproducible, not just
"roughly similar."

There's also clamping/padding logic: if a pool is too small to satisfy the
requested split (e.g. a question has only 2 positive docs but the math wants
3), the shortfall is backfilled from the other pool so every call still
returns exactly `passage_num` documents.

`sample_records(records, sample_size)` does a single seeded subsample of
*which questions* to evaluate (independent of the per-question doc sampling
above) — this is the `--sample-size` CLI flag.

### 4.3 `prompts.py` — the exact evaluation contract
`SYSTEM_INSTRUCTION` and `USER_TEMPLATE` are transcribed verbatim from
Figure 3 of the paper — this is deliberate: the metrics only work if the model
is asked, in the same way the paper asked it, to (a) answer from the docs, (b)
emit the exact refusal sentence when nothing supports an answer, (c) emit the
exact factual-error sentence when it detects a contradiction. Everything
downstream (`matching.py`) does substring matching against these literal
sentences, so the wording here is load-bearing, not decorative.

The same file also holds the **mitigation prompt variants** (one constant/
template per mitigation), kept alongside the baseline prompt so the diff
between baseline and mitigated behavior is easy to review side-by-side:
- `VERIFY_THEN_ANSWER_SYSTEM` — baseline instruction + an explicit "check
  first, then refuse-only-if-nothing-supports-it" clause.
- `DECOMPOSE_SYSTEM` / `MERGE_SYSTEM` — two separate system prompts for the
  split→answer→merge pipeline (see §4.7).
- `PARAMETRIC_ONLY_SYSTEM` / `CROSSCHECK_SYSTEM` / `CROSSCHECK_USER_TEMPLATE`
  — for the parametric-knowledge-vs-retrieved-docs cross-check (see §4.8).

### 4.4 `llm_clients.py` — unified client + caching + resilience
`BaseLLMClient` is an ABC with one abstract method, `_call(system, user,
temperature) -> str`. Three concrete subclasses (`GeminiClient`, `GroqClient`,
`MistralClient`) each wrap their SDK's chat/generate call. The public
entrypoint is `generate()`, which wraps `_call()` with:

1. **Disk caching** — `_cache_key()` hashes `sha256(name|model_id|temperature|system|user)`
   and looks up `cache/<hash>.json`. Cache hit → return immediately, no API
   call, no rate-limit consumption. Cache miss → call the API, then persist
   `{model, model_id, system, user, response}` to disk. This is why the repo
   already has ~280 cached responses even without a fresh full run, and why
   `aggregate_results.py` and repeated experiment runs are "free": once a
   (model, ability, mitigation, exact-prompt) combination has been seen once,
   it never touches the network again.
2. **Throttling** — `_throttle()` enforces `min_interval_seconds` between real
   API calls per client instance (Gemini free tier ≈ 5 req/min → 13s;
   Groq/Mistral looser but still capped). This exists because free-tier APIs
   return 429s under bursty calling patterns, especially across a 3-model ×
   4-ability × 2-condition grid.
3. **Retry with backoff** — `@retry(wait=wait_random_exponential(...),
   stop=stop_after_attempt(...))` from `tenacity` wraps every `_call()`, so
   transient 429/5xx errors self-heal instead of crashing a long batch run.

`get_client(name)` is a tiny factory/registry (`CLIENT_REGISTRY`) so
`experiments/*.py` and `run_experiment.py` never import a concrete client
class directly — swapping in a 4th model later means adding one subclass +
one registry entry, nothing else changes.

### 4.5 `matching.py` — turning free text into booleans
LLM responses are unstructured text, not JSON, so this module is the bridge
between "raw model output" and "countable metric." Three checks:
- `contains_answer(response, answer)` — normalizes both strings (lowercase,
  strip punctuation, collapse whitespace) and checks that **every** alias
  group in `answer` has at least one alias present as a substring. `answer`'s
  shape varies by dataset: a plain string (counterfactual), a single alias
  list, or (for multi-hop `en_int.json` questions) a list of alias lists —
  `_alias_lists()` normalizes all three shapes before matching, and
  `contains_answer` requires all sub-answer groups to be present (this is why
  it's the right metric for Information Integration: partial credit for
  answering only one sub-question doesn't count as a correct compound answer).
- `is_refusal(response)` — normalized substring match against the exact
  `REFUSAL_PHRASE` from `config.py`.
- `flags_factual_error(response)` — same idea, against `FACTUAL_ERROR_PHRASE`.

This is intentionally simple (no embeddings, no LLM-as-judge) — it mirrors
the paper's own exact-match `Rej`/`ED`/`CR` definitions, and explicitly
excludes the paper's LLM-judged `Rej*`/`ED*` variants (per the task PDF's
instruction: "you need not report Rej*/ED*"), so results stay deterministic
and free to compute.

### 4.6 `metrics.py` — the arithmetic, isolated from everything else
Four pure functions, each taking a `list[bool]` (already computed by
`matching.py`) and returning a fraction in `[0, 1]`:
- `accuracy(results)` — Noise Robustness & Information Integration.
- `rejection_rate(refusals)` — Negative Rejection.
- `error_detection_rate(flags)` — Counterfactual Robustness (ED).
- `error_correction_rate(flags, corrected)` — **conditional** on detection:
  denominator is only the subset where an error was actually flagged, not the
  full sample — this matches the paper's CR definition (correction is only
  meaningful once detection happened).

Kept deliberately dependency-free and side-effect-free so they're trivial to
unit test in isolation from any API/network concern.

### 4.7 `experiments/*.py` — one orchestrator per ability
Each file follows the same shape: `run(client, sample_size, mitigation) ->
list[dict]`. Concretely:
- **`noise_robustness.py`** loops `config.NOISE_RATES`, for each rate builds a
  fresh noisy context per record via `sample_docs`, optionally passes it
  through `noise_filter.filter_relevant_docs`, prompts, scores with
  `contains_answer`, and emits one summary row per noise rate.
- **`negative_rejection.py`** fixes `noise_rate=1.0` (i.e., **all 5 docs are
  noise, zero positive docs** — the negative-rejection testbed by
  construction), and either calls the baseline prompt or
  `decision_calibration.verify_then_answer`.
- **`information_integration.py`** loops `config.INTEGRATION_NOISE_RATES`
  (only 0/0.2/0.4 — matching the paper's narrower Table 5 range for this
  ability), and either uses the baseline prompt or
  `decompose_integrate.decompose_and_answer`.
- **`counterfactual_robustness.py`** reads `rec.extra["positive_wrong"]`
  (falling back to `rec.positive`) — this is the counterfactual document
  containing a plausible-but-wrong answer — and either baseline-prompts or
  calls `fact_check.cross_check_answer`. It scores **both** `flags_factual_error`
  (ED) and `contains_answer` against the *true* answer (CR), in the same pass.

Every row dict includes `ability`, `model`, `mitigation` (bool), and the
relevant noise_rate/metric columns — this flat schema is exactly what lets
`aggregate_results.py` `pd.concat()` every CSV in `results/` and pivot on
`(model, mitigation)` without any per-ability special-casing beyond which
columns to pivot on.

### 4.8 `mitigations/*.py` — the "how we improved it" layer
Each mitigation is a small, explainable technique targeting the paper's own
stated root cause for that ability's failures (see README §4 table for the
paper-quote-to-technique mapping). Implementation notes:
- **`noise_filter.filter_relevant_docs`** — zero extra LLM calls. Tokenizes
  the query (stopword-filtered), scores every doc by lexical-overlap word
  count with the query, keeps the top `keep_fraction=0.6` of docs (stable
  sort preserves original relative order among ties), always keeps ≥1 doc.
  This is a classic "cheap retrieval re-ranking" trick applied *after*
  sampling but *before* the answering prompt — it shrinks the effective noise
  ratio the model has to reason over without touching the LLM call budget.
- **`decision_calibration.verify_then_answer`** — one LLM call, but with a
  stricter system prompt (`VERIFY_THEN_ANSWER_SYSTEM`) that adds an explicit
  "check first, refuse-only-if-nothing-supports-it" instruction on top of the
  baseline Figure-3 instruction.
- **`decompose_integrate.decompose_and_answer`** — **multiple** LLM calls: (1)
  decompose the question into sub-questions, (2) answer each sub-question
  independently against the *same* document set, (3) merge all sub-answers
  into one final answer via a dedicated merge prompt. Trades latency/cost for
  accuracy — the README explicitly notes the paper itself flags this
  latency/accuracy tradeoff.
- **`fact_check.cross_check_answer`** — two LLM calls: (1) ask the model to
  answer using **only its own parametric knowledge** (no documents at all),
  (2) feed that answer alongside the retrieved docs into a cross-check prompt
  that explicitly asks the model to trust its own knowledge unless the docs
  provide strong independent evidence otherwise, and to flag+correct any
  conflict. This directly operationalizes the paper's diagnosis that models
  "heavily depend on the information they retrieve" even when they already
  know better.

### 4.9 `scripts/run_experiment.py` and `scripts/aggregate_results.py`
`run_experiment.py` is the CLI surface: `--ability {noise_robustness,
negative_rejection, information_integration, counterfactual_robustness, all}`
× `--model {gemini, groq, mistral, all}` × `--sample-size N` × `--mitigation`
(flag). It resolves a client once per model (reused across abilities to
respect the client's own throttle state), calls the matching
`ABILITY_RUNNERS[...].run(...)`, flattens all rows into one `pandas.DataFrame`,
and writes `results/results_<abilities>_<models>_<baseline|mitigated>.csv`.

`aggregate_results.py` globs every `results/results_*.csv`, concatenates them,
and builds 4 pivot tables (one per ability) — e.g. Noise Robustness pivots
`noise_rate` (rows) × `(model, mitigation)` (columns) → `accuracy` (values),
which is exactly the shape needed to drop straight into a paper-style
table or a line chart. Each pivot is also saved to `results/summary_<ability>.csv`.

---

## 5. Why 3 non-OpenAI/non-ChatGLM models?
The task explicitly asks for "at least 3 different LLMs," and the reference
repo/paper already benchmarked GPT-3.5 and ChatGLM — so this project
deliberately picked 3 **free-tier** alternatives to avoid re-treading the
paper's own model choices and to keep the whole grid runnable without a paid
API budget: **Gemini Flash** (Google AI Studio), **Llama 3.1 8B Instant**
(Groq — fast open-weight inference), **Mistral Small** (Mistral La
Plateforme). All three are wired through the exact same `BaseLLMClient`
interface, so every experiment/mitigation function is model-agnostic — no
`if model == "gemini"` branching anywhere outside `llm_clients.py`.

---

## 6. What "independent re-implementation" means concretely
Per the task's explicit instruction ("do not copy paste the implementation
code, use it only for reference"), the reference repo
(github.com/chen700564/RGB) was used **only** to understand the JSON schema
(`id/query/answer/positive/negative` fields, the nested list-of-lists shape
for multi-hop `positive` in `en_int.json`, the `positive_wrong` field for
counterfactual docs). Everything else — the sampling math in `data_loader.py`,
the matching logic in `matching.py`, the metric formulas in `metrics.py`, the
client wrapper/cache/retry design in `llm_clients.py`, and all 4 mitigation
techniques — is original code written against the paper's Figure 3 prompt and
metric definitions, not ported from the reference implementation.

---

## 7. How to trace a single example end-to-end (debugging recipe)
If a number in the deck looks wrong, here's how to trace it by hand:
1. Pick `ability` + `model` + `mitigation` → find the row in `results/results_*.csv`.
2. Re-derive the exact document set: `data_loader.sample_docs(record, noise_rate,
   passage_num=5)` with the same `record.id`/`noise_rate`/`config.RANDOM_SEED`
   is deterministic — you'll get the identical docs used in the real run.
3. Re-build the prompt: `prompts.build_prompt(record.query, docs)` (or the
   relevant mitigation function) reproduces the exact system/user strings sent.
4. Look up `cache/<sha256(...)>.json` (same hash formula as
   `BaseLLMClient._cache_key`) to see the raw model response that was
   actually scored — no need to re-call the API.
5. Run `matching.contains_answer(response, record.answer)` /
   `is_refusal(response)` / `flags_factual_error(response)` directly in a
   REPL to see exactly why it was scored true/false.

---

## 8. Known gaps / honest caveats (for code review)
- `tests/` currently only contains `__init__.py` — no unit tests have been
  written yet for `matching.py`/`metrics.py`/`data_loader.py`, even though
  they're pure functions and the easiest/cheapest things in the repo to test
  (no network, no API keys needed). This is the top item to pick up next.
- `run_experiment.py` re-runs the **full** `ABILITY_RUNNERS[ability]` loop
  regardless of what's already cached — the caching happens at the
  `llm_clients.generate()` layer (skips the network call), not at the
  experiment layer (still re-does sampling/matching/metric math), which is
  fine cost-wise but means a "dry" re-run still takes real wall-clock time
  proportional to sample size.
- Mitigations for Information Integration and Counterfactual Robustness cost
  2-3x the LLM calls of baseline (decompose+merge; parametric+crosscheck) —
  worth calling out explicitly if anyone asks about "fair comparison" cost.
