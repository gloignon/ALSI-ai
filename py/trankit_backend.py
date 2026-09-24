"""Trankit backend for ALSI parse_text().

Loaded once via reticulate::source_python(); thereafter called through
reticulate::py$trankit_parse_texts().

The pipeline is initialised lazily on first call and cached in the module-level
_pipeline variable so it is not reloaded on every call.
"""

import io
import sys

_SKIP = ("Active language:", "=" * 10)


class _FilteredStderr(io.TextIOBase):
    """Forwards stdout to stderr, dropping Trankit's internal banner lines."""
    def write(self, s):
        if s and not any(s.startswith(p) for p in _SKIP) and s.strip():
            print(s.rstrip(), file=sys.stderr, flush=True)
        return len(s)


_pipeline = None
_model_dir = None
_category = None


def trankit_load(model_dir: str, category: str = "customized-mwt", gpu: bool = False):
    """Load (or reload) the Trankit pipeline. Safe to call multiple times."""
    global _pipeline, _model_dir, _category

    if _pipeline is not None and _model_dir == model_dir and _category == category:
        return  # already loaded

    from trankit import Pipeline
    print("Loading ALSI French model (Trankit) …", file=sys.stderr, flush=True)
    import contextlib
    with contextlib.redirect_stdout(_FilteredStderr()):
        _pipeline = Pipeline(lang=category, cache_dir=model_dir, gpu=gpu)
    _model_dir = model_dir
    _category = category
    print("Trankit pipeline ready.", file=sys.stderr, flush=True)


def trankit_parse_texts(texts, doc_ids, model_dir: str,
                        category: str = "customized-mwt", gpu: bool = False):
    """Parse a list of texts and return a list of token-level dicts.

    Parameters
    ----------
    texts    : list[str]  — one entry per document
    doc_ids  : list[str]  — parallel doc identifiers
    model_dir: str        — path to trankit_fr_v1 (or equivalent)
    category : str        — category name used at training time
    gpu      : bool       — whether to use GPU for inference

    Returns
    -------
    list[dict] — one dict per token with keys matching udpipe output columns:
        doc_id, sentence_id, token_id, token, lemma, upos, xpos,
        feats, head_token_id, dep_rel
    """
    trankit_load(model_dir, category, gpu)

    rows = []
    for doc_id, text in zip(doc_ids, texts):
        if not text or not text.strip():
            continue

        doc_result = _pipeline(text)
        global_sent_id = 0

        for sent in doc_result.get("sentences", []):
            global_sent_id += 1
            tokens = sent.get("tokens", [])
            local_id = 0

            for tok in tokens:
                # MWT tokens have an 'expanded' list — iterate over subtokens
                if "expanded" in tok:
                    for sub in tok["expanded"]:
                        local_id += 1
                        rows.append(_make_row(doc_id, global_sent_id, local_id, sub))
                else:
                    local_id += 1
                    rows.append(_make_row(doc_id, global_sent_id, local_id, tok))

    return rows


def _make_row(doc_id, sentence_id, token_id, tok):
    head_raw = tok.get("head", 0)
    # Trankit uses 0 for root; keep as 0 to match udpipe convention
    return {
        "doc_id":        str(doc_id),
        "sentence_id":   int(sentence_id),
        "token_id":      int(token_id),
        "token":         str(tok.get("text", "")),
        "lemma":         str(tok.get("lemma", "")),
        "upos":          str(tok.get("upos", "")),
        "xpos":          str(tok.get("xpos", "")),
        "feats":         str(tok.get("feats", "")) if tok.get("feats") else NA_STRING,
        "head_token_id": int(head_raw),
        "dep_rel":       str(tok.get("deprel", "")),
    }


# Sentinel for missing values — R will coerce "NA" strings to NA
NA_STRING = "NA"
