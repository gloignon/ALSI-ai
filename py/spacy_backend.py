"""spaCy backend for ALSI parse_text().

Loaded once via reticulate::source_python(); thereafter called through
reticulate::py$spacy_parse_texts().

The pipeline is initialised lazily on first call and cached in the module-level
_nlp variable so it is not reloaded on every call.

Written from scratch; the sentence-relative token ID idiom (sent_start = sent[0].i)
and the lazy-load/cache pattern were informed by the cleanNLP R package
(Taylor Arnold, https://github.com/taylor-arnold/rpkg/tree/master/cleanNLP),
which is acknowledged here as a design reference.
"""

import sys

_nlp = None
_model_path = None


def spacy_load(model_path: str):
    """Load (or reload) the spaCy pipeline. Safe to call multiple times."""
    global _nlp, _model_path

    if _nlp is not None and _model_path == model_path:
        return

    import spacy
    print(f"Loading ALSI French model (spaCy) from {model_path} …",
          file=sys.stderr, flush=True)
    _nlp = spacy.load(model_path)
    _model_path = model_path
    print("spaCy pipeline ready.", file=sys.stderr, flush=True)


def spacy_parse_texts(texts, doc_ids, model_path: str):
    """Parse a list of texts and return a list of token-level dicts.

    Parameters
    ----------
    texts      : list[str]  — one entry per document
    doc_ids    : list[str]  — parallel doc identifiers
    model_path : str        — path to the spaCy model directory

    Returns
    -------
    list[dict] — one dict per token with keys matching udpipe output columns:
        doc_id, sentence_id, token_id, token, lemma, upos, xpos,
        feats, head_token_id, dep_rel
    """
    spacy_load(model_path)

    rows = []
    for doc_id, text in zip(doc_ids, texts):
        if not text or not text.strip():
            continue

        doc = _nlp(text)
        for sent_id, sent in enumerate(doc.sents, start=1):
            sent_start = sent[0].i
            for tok in sent:
                token_id = tok.i - sent_start + 1
                if tok.dep_.upper() == "ROOT":
                    head_token_id = 0
                else:
                    head_token_id = tok.head.i - sent_start + 1
                feats = str(tok.morph) if tok.morph else NA_STRING
                rows.append({
                    "doc_id":        str(doc_id),
                    "sentence_id":   int(sent_id),
                    "token_id":      int(token_id),
                    "token":         tok.text,
                    "lemma":         tok.lemma_,
                    "upos":          tok.tag_,
                    "xpos":          NA_STRING,
                    "feats":         feats,
                    "head_token_id": int(head_token_id),
                    "dep_rel":       tok.dep_,
                })

    return rows


# Sentinel for missing values — R will coerce "NA" strings to NA
NA_STRING = "NA"
