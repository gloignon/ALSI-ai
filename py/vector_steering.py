"""Unsupervised text complexity classifier using pretrained steering vectors.

Projects text onto a complexity direction extracted via the double-dip method
(Wiki-Vikidia contrastive pairs). Returns a scalar score per text.

Designed for use from R via reticulate:

    library(reticulate)
    clf <- import("src.classifier")
    scorer <- clf$load_scorer("Lajavaness/sentence-camembert-base")
    result <- clf$score(scorer, "Le chat dort sur le tapis.")
    results <- clf$score_batch(scorer, c("text1", "text2"))
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# ── Model registry ──────────────────────────────────────────────
# Tested defaults from layer sweeps (see DOUBLEDIP.md).

MODEL_DEFAULTS = {
    "Lajavaness/sentence-camembert-base": {
        "type": "encoder",
        "pool": "mean",
        "inject_layer": 3,
        "extract_layer": 10,
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "type": "encoder",
        "pool": "mean",
        "inject_layer": 1,
        "extract_layer": 1,
    },
    "lightonai/pagnol-small": {
        "type": "causal",
        "pool": "last",
        "inject_layer": 6,
        "extract_layer": 10,
    },
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _auto_device():
    """Pick the best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Internal helpers ────────────────────────────────────────────

def _get_layers(model, model_type):
    """Return the transformer layer ModuleList."""
    if model_type == "encoder":
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer
        if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
            return model.transformer.layer
    else:
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return model.gpt_neox.layers
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")


def _extract(model, tokenizer, text, layer_modules, extract_layer, pool, device):
    """Single forward pass, capture hidden state at extract_layer, pool to 1-D."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    captured = {}

    def hook(_, __, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    handle = layer_modules[extract_layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        handle.remove()

    if "h" not in captured:
        return None

    h = captured["h"][0]  # [seq_len, hidden_dim]
    if pool == "mean":
        mask = enc["attention_mask"][0].unsqueeze(-1).to(h.device)
        return ((h * mask).sum(0) / mask.sum()).cpu()
    return h[-1, :].cpu()


def _split_sentences(text):
    """Split text into sentences using basic French/English punctuation rules."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in parts if len(s.split()) >= 3]


# ── Public API ──────────────────────────────────────────────────

def load_scorer(model_name="Lajavaness/sentence-camembert-base",
                vector_path=None,
                inject_layer=None,
                extract_layer=None,
                pool=None,
                device=None,
                reference_path=None):
    """Load a model and steering vector for complexity scoring.

    Args:
        model_name: HuggingFace model ID. If in MODEL_DEFAULTS, layers/pool
                    are set automatically.
        vector_path: Path to .npz file with the steering vector. If None,
                     looks for a default vector in out/ndip/.
        inject_layer: Override injection layer (for building vectors only).
        extract_layer: Override extraction layer.
        pool: Override pooling strategy ("mean" or "last").
        device: Torch device.
        reference_path: Path to .npz with reference distribution for percentile
                        calibration. If None, raw projections only.

    Returns:
        Opaque scorer dict to pass to score() / score_batch().
    """
    if device is None:
        device = _auto_device()

    # Resolve defaults
    if model_name in MODEL_DEFAULTS:
        defaults = MODEL_DEFAULTS[model_name]
        model_type = defaults["type"]
        if pool is None:
            pool = defaults["pool"]
        if extract_layer is None:
            extract_layer = defaults["extract_layer"]
        if inject_layer is None:
            inject_layer = defaults["inject_layer"]
    else:
        model_type = None
        if pool is None or extract_layer is None:
            raise ValueError(
                f"Model '{model_name}' not in MODEL_DEFAULTS. "
                f"Please specify pool, inject_layer, and extract_layer explicitly."
            )

    # Infer model type if not from registry
    if model_type is None:
        model_type = "encoder"  # assume encoder if unknown

    # Load model
    if model_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    else:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    layer_modules = _get_layers(model, model_type)

    # Load vector
    if vector_path is None:
        slug = model_name.replace("/", "_")
        vector_path = _PROJECT_ROOT / f"out/ndip/{slug}/V2_L{extract_layer}.npz"
    else:
        vector_path = Path(vector_path)

    if not vector_path.exists():
        raise FileNotFoundError(
            f"Steering vector not found at {vector_path}. "
            f"Run experiment_ndip.py first to generate it, or specify vector_path."
        )

    data = np.load(str(vector_path))
    vec_key = str(extract_layer)
    if vec_key not in data:
        # Try first available key
        vec_key = list(data.keys())[0]
    vector = torch.from_numpy(data[vec_key]).float()
    unit_vec = vector / (vector.norm() + 1e-8)

    # Load reference distribution for calibration
    ref_dist = None
    if reference_path is not None:
        ref_data = np.load(str(reference_path))
        ref_dist = np.sort(ref_data["projections"])

    return {
        "model": model,
        "tokenizer": tokenizer,
        "layer_modules": layer_modules,
        "extract_layer": extract_layer,
        "pool": pool,
        "device": device,
        "unit_vec": unit_vec,
        "ref_dist": ref_dist,
        "model_name": model_name,
    }


def score(scorer, text):
    """Score a single text for complexity.

    Args:
        scorer: Scorer dict from load_scorer().
        text: String to score.

    Returns:
        Dict with keys: text, projection, percentile (if calibrated), label.
    """
    h = _extract(
        scorer["model"], scorer["tokenizer"], text,
        scorer["layer_modules"], scorer["extract_layer"],
        scorer["pool"], scorer["device"],
    )
    if h is None:
        return {"text": text, "projection": float("nan"),
                "percentile": None, "label": None}

    proj = float(torch.dot(h, scorer["unit_vec"]))
    result = {"text": text, "projection": proj}

    ref = scorer["ref_dist"]
    if ref is not None:
        pct = float(np.searchsorted(ref, proj) / len(ref))
        result["percentile"] = pct
        if pct < 1/3:
            result["label"] = "simple"
        elif pct < 2/3:
            result["label"] = "intermediate"
        else:
            result["label"] = "complex"
    else:
        result["percentile"] = None
        result["label"] = None

    return result


def score_batch(scorer, texts):
    """Score a list of texts.

    Args:
        scorer: Scorer dict from load_scorer().
        texts: List of strings.

    Returns:
        List of dicts (same format as score()). Converts to data.frame in R.
    """
    return [score(scorer, t) for t in texts]


def score_document(scorer, text, split=True):
    """Score a document, optionally splitting into sentences.

    Args:
        scorer: Scorer dict from load_scorer().
        text: Document text.
        split: If True, split into sentences and score each. If False,
               score the whole text as one input.

    Returns:
        Dict with keys: sentences (list of per-sentence results),
        mean_projection, percentile, label.
    """
    if not split:
        return score(scorer, text)

    sentences = _split_sentences(text)
    if not sentences:
        return {"sentences": [], "mean_projection": float("nan"),
                "percentile": None, "label": None}

    results = score_batch(scorer, sentences)
    projs = [r["projection"] for r in results if not np.isnan(r["projection"])]

    if not projs:
        return {"sentences": results, "mean_projection": float("nan"),
                "percentile": None, "label": None}

    mean_proj = float(np.mean(projs))
    out = {"sentences": results, "mean_projection": mean_proj}

    ref = scorer["ref_dist"]
    if ref is not None:
        pct = float(np.searchsorted(ref, mean_proj) / len(ref))
        out["percentile"] = pct
        if pct < 1/3:
            out["label"] = "simple"
        elif pct < 2/3:
            out["label"] = "intermediate"
        else:
            out["label"] = "complex"
    else:
        out["percentile"] = None
        out["label"] = None

    return out


def build_reference(scorer, texts, save_path=None):
    """Build a reference distribution from a list of texts.

    Project each text onto the complexity direction and save the sorted
    array of projections. Use the output path with load_scorer(reference_path=...).

    Args:
        scorer: Scorer dict from load_scorer().
        texts: List of texts (e.g. all Wiki + Vikidia texts).
        save_path: Where to save the .npz. If None, returns the array.

    Returns:
        Sorted numpy array of projections.
    """
    projs = []
    for t in texts:
        r = score(scorer, t)
        if not np.isnan(r["projection"]):
            projs.append(r["projection"])
    ref = np.sort(np.array(projs))

    if save_path is not None:
        np.savez(str(save_path), projections=ref)
        print(f"Reference distribution saved to {save_path} ({len(ref)} texts)")

    return ref
