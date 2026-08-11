"""Constants shared across the RGB evaluation pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

CACHE_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DATA_FILES = {
    "noise_robustness": DATA_DIR / "en_refine.json",
    "negative_rejection": DATA_DIR / "en_refine.json",
    "information_integration": DATA_DIR / "en_int.json",
    "counterfactual_robustness": DATA_DIR / "en_fact.json",
}

# Paper default: 5 documents shown to the model per question.
PASSAGE_NUM = 5

# Noise ratios used for the Noise Robustness testbed (paper's Table 1 values).
NOISE_RATES = [0.0, 0.2, 0.4, 0.6, 0.8]

# Negative Rejection is evaluated at noise_rate = 1.0 (only noise docs, no positives).
NEGATIVE_REJECTION_NOISE_RATE = 1.0

# Information Integration noise ratios used in the paper's Table 5.
INTEGRATION_NOISE_RATES = [0.0, 0.2, 0.4]

# Exact phrases from Figure 3 of the paper -- used for matching, not generation.
REFUSAL_PHRASE = "I can not answer the question because of the insufficient information in documents."
FACTUAL_ERROR_PHRASE = "There are factual errors in the provided documents."

# Model registry: logical name -> concrete API model id.
# All three are free-tier, non-OpenAI/ChatGPT/ChatGLM models (per task instructions).
MODELS = {
    "gemini": "gemini-flash-latest",
    "groq": "llama-3.1-8b-instant",
    "mistral": "mistral-small-latest",
}

RANDOM_SEED = 42
