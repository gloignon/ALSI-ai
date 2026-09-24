import math
import os
import warnings
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

LN2 = math.log(2.0)

# Module-level model cache. reticulate::source_python() re-executes this
# file on every call from the R API; globals().get() keeps an already-loaded
# model instead of resetting the cache to None.
_tokenizer = globals().get("_tokenizer")
_model = globals().get("_model")
_device = globals().get("_device")
_mode = globals().get("_mode")
_model_name = globals().get("_model_name")
_add_prefix_space = globals().get("_add_prefix_space")


def _best_device():
    if os.environ.get("ALSIO_FORCE_CPU") in ("1", "true", "TRUE", "yes", "YES"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_llm_model(
    model_name: str,
    mode: str = "mlm",
    use_fast: bool = True,
    trust_remote_code: bool = True,
    add_prefix_space: Optional[bool] = None,
    force_fast: bool = False,
):
    """Load tokenizer + model.

    `add_prefix_space=None` (default) resolves to `True` for `mode="ar"` and
    `False` for `mode="mlm"`. Byte-level BPE tokenizers (GPT-2-family, used by
    most French AR models) require `add_prefix_space=True` when called with
    `is_split_into_words=True`, otherwise pre-split words are encoded without
    their leading-space marker and land on the wrong sub-distribution (e.g.
    "chat" vs "Ġchat"). SentencePiece tokenizers (CamemBERT/Flaubert) are
    unaffected by this flag.

    `force_fast=True` bypasses AutoTokenizer entirely and loads
    PreTrainedTokenizerFast directly from tokenizer.json. Use this when a
    repo's tokenizer_config.json points at a slow tokenizer class whose
    merges/vocab files are missing (e.g., almanach/camembertv2-base ships only
    tokenizer.json but declares tokenizer_class: RobertaTokenizer, which then
    falls through to character-level tokenization).

    `force_fast=False` (default) tries AutoTokenizer and auto-falls-back to
    PreTrainedTokenizerFast only if the loaded tokenizer breaks the French word
    "Cette" into more than 3 subwords (a clear sign of the BPE-merges-not-loaded
    failure mode).
    """
    global _tokenizer, _model, _device, _mode, _model_name, _add_prefix_space
    _device = _best_device()
    if add_prefix_space is None:
        add_prefix_space = (mode == "ar")
    tokenizer_kwargs = {
        "trust_remote_code": trust_remote_code,
        "use_fast": use_fast,
        "add_prefix_space": add_prefix_space,
    }

    if force_fast:
        from transformers import PreTrainedTokenizerFast
        _tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_name, add_prefix_space=add_prefix_space
        )
    else:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        except TypeError:
            tokenizer_kwargs.pop("add_prefix_space", None)
            _tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        # Auto-fallback: detect character-level BPE failure and switch to
        # PreTrainedTokenizerFast (reads tokenizer.json directly).
        try:
            _probe_ids = _tokenizer("Cette", add_special_tokens=False)["input_ids"]
            if len(_probe_ids) > 3:
                from transformers import PreTrainedTokenizerFast
                _fast = PreTrainedTokenizerFast.from_pretrained(
                    model_name, add_prefix_space=add_prefix_space
                )
                _probe_fast = _fast("Cette", add_special_tokens=False)["input_ids"]
                if len(_probe_fast) <= 3:
                    _tokenizer = _fast
        except Exception:
            pass

    if mode == "mlm":
        _model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    elif mode == "ar":
        _model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    else:
        raise ValueError("mode must be one of: 'mlm', 'ar'")
    _mode = mode
    _model_name = model_name
    _add_prefix_space = bool(add_prefix_space)
    _model = _model.to(_device)
    _model.eval()
    print(f"Loaded {mode.upper()} model: {model_name} on {_device}")
    return True


def get_llm_state():
    return {"model_name": _model_name, "mode": _mode, "add_prefix_space": _add_prefix_space}


def _compute_surprisal_entropy(logits: torch.Tensor, target_id: int, temperature: float = 1.0,
                               compute_entropy: bool = True):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = logits / temperature
    log_probs = torch.log_softmax(scaled, dim=-1)
    surprisal = -log_probs[target_id].item() / LN2
    if not compute_entropy:
        # entropy == "none": skip the full-vocab entropy sum entirely.
        return surprisal, float("nan")
    probs = torch.softmax(scaled, dim=-1)
    entropy = -(probs * log_probs).sum().item() / LN2
    return surprisal, entropy


def _topk_predictions(logits: torch.Tensor, k: int, temperature: float = 1.0):
    """Top-k tokens and probabilities from one position's logits.

    Returns (tokens, probs): the k most probable vocabulary items decoded to
    readable strings, and their softmax probabilities (same temperature
    scaling as the surprisal computation)."""
    probs = torch.softmax(logits / temperature, dim=-1)
    top = torch.topk(probs, k)
    ids = top.indices.tolist()
    tokens = []
    for tid in ids:
        text = _tokenizer.decode([tid]).strip()
        if not text:
            text = _tokenizer.convert_ids_to_tokens([tid])[0]
        tokens.append(text)
    return tokens, [float(p) for p in top.values.tolist()]


def _get_word_ids(encoding) -> Optional[List]:
    """Try to get word_ids from encoding; return None if unavailable."""
    if hasattr(encoding, "word_ids"):
        try:
            return encoding.word_ids(0)
        except Exception:
            return None
    return None


_ENTROPY_MODES = ("mean", "sum", "onset", "offset", "none")


def _aggregate_to_words(
    word_ids: List, n_words: int, context_len: int,
    subword_surprisals: List[float], subword_entropies: List[float],
    entropy: str = "onset",
) -> Dict[str, List]:
    """Aggregate subword scores to word level using word_ids mapping.

    `entropy` controls how a multi-subword word's entropy is summarized:
      - "onset": entropy of the word's first subtoken (default).
      - "mean": mean of subword entropies.
      - "sum": sum of subword entropies.
      - "offset": entropy of the word's last subtoken.
      - "none": skip entropy entirely; every word entropy is reported as NaN.
    """
    word_surps = [0.0] * n_words
    word_ents = [0.0] * n_words
    word_onset_ents = [float("nan")] * n_words
    word_offset_ents = [float("nan")] * n_words
    word_counts = [0] * n_words
    # Track surprisal and entropy validity separately: when entropy is skipped
    # (entropy == "none") its NaNs must not poison an otherwise valid surprisal.
    word_surp_has_nan = [False] * n_words
    word_ent_has_nan = [False] * n_words

    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if 0 <= wid < n_words:
            sub_pos = context_len + pos
            surp = subword_surprisals[sub_pos]
            ent = subword_entropies[sub_pos]
            if math.isnan(surp):
                word_surp_has_nan[wid] = True
            else:
                word_surps[wid] += surp
            if math.isnan(ent):
                word_ent_has_nan[wid] = True
            else:
                word_ents[wid] += ent
                if math.isnan(word_onset_ents[wid]):
                    word_onset_ents[wid] = ent
                word_offset_ents[wid] = ent
            word_counts[wid] += 1

    # Reject an unknown mode once, up front, so a requested mode is always the
    # one actually applied (never a silent fallback) and the error does not
    # depend on the data having a clean word to reach an else-branch.
    if entropy not in _ENTROPY_MODES:
        raise ValueError(
            f"unknown entropy {entropy!r}; expected one of {_ENTROPY_MODES}"
        )

    # Surprisal: sum of subword surprisals (chain rule of probability).
    # Entropy: summarized across subwords according to `entropy` — a
    # heuristic, not the entropy of any single distribution, but a useful
    # per-word summary.
    #
    # NaN handling depends on whether entropy is being computed:
    #  - For a computing mode ("onset"/"mean"/"sum"/"offset"), a word with any
    #    NaN subword (surprisal OR entropy) is reported NaN for BOTH metrics,
    #    rather than a partial aggregate that would understate its score.
    #  - For "none", entropy is deliberately skipped, so its (all-NaN) entropy
    #    array must NOT poison an otherwise valid surprisal.
    for i in range(n_words):
        if entropy == "none":
            word_ents[i] = float("nan")
            if not (word_counts[i] > 0 and not word_surp_has_nan[i]):
                word_surps[i] = float("nan")
            continue

        word_has_nan = word_surp_has_nan[i] or word_ent_has_nan[i]
        if word_counts[i] > 0 and not word_has_nan:
            if entropy == "onset":
                word_ents[i] = word_onset_ents[i]
            elif entropy == "offset":
                word_ents[i] = word_offset_ents[i]
            elif entropy == "sum":
                pass  # word_ents[i] already holds the sum
            else:  # "mean"
                word_ents[i] = word_ents[i] / word_counts[i]
        else:
            word_surps[i] = float("nan")
            word_ents[i] = float("nan")

    return {
        "word_surprisals": word_surps,
        "word_entropies": word_ents,
        "word_token_counts": word_counts,
    }


def _empty_result() -> Dict:
    return {
        "word_surprisals": [],
        "word_entropies": [],
        "word_token_counts": [],
        "word_tokens": [],
        "word_oov": [],
        "subword_surprisals": [],
        "subword_entropies": [],
        "subword_word_ids": [],
        "subword_tokens": [],
    }


_UNK_STRINGS = {"[UNK]", "<unk>"}


def _extract_word_tokens(
    is_split_into_words: bool,
    input_obj,
    encoding,
    word_ids,
    sentence_ids,
    n_words: int,
) -> tuple:
    """Return (tokens, oov_flags) — one entry per word.

    When a word maps entirely to UNK subwords, the original surface form is
    recovered instead of '[UNK]' and the corresponding `oov_flags` entry is
    set to True. For pre-split input the original word string is used
    directly; for raw-string input the word's character span is recovered via
    `encoding.token_to_chars()` and sliced from `input_obj`.
    """
    oov_template = [False] * n_words

    if is_split_into_words:
        words = list(input_obj)
        # Check each word: if its tokenization is a single UNK, flag it.
        oov = list(oov_template)
        for i, w in enumerate(words):
            ids = _tokenizer(w, add_special_tokens=False)["input_ids"]
            toks = _tokenizer.convert_ids_to_tokens(ids)
            if toks and all(t in _UNK_STRINGS for t in toks):
                oov[i] = True
        return words, oov

    if word_ids is None or n_words == 0:
        return [], []

    subword_tokens = _tokenizer.convert_ids_to_tokens(sentence_ids.tolist())
    groups: Dict[int, List[int]] = {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        groups.setdefault(wid, []).append(pos)

    out = [""] * n_words
    oov = list(oov_template)
    for wid, positions in groups.items():
        if 0 <= wid < n_words:
            subs = [subword_tokens[p] for p in positions]
            all_unk = all(t in _UNK_STRINGS for t in subs)
            if all_unk and isinstance(input_obj, str):
                spans = [encoding.token_to_chars(p) for p in positions]
                spans = [s for s in spans if s is not None]
                if spans:
                    out[wid] = input_obj[min(s.start for s in spans):max(s.end for s in spans)]
                else:
                    out[wid] = "".join(subs)
            else:
                try:
                    out[wid] = _tokenizer.convert_tokens_to_string(subs).strip()
                except Exception:
                    out[wid] = "".join(subs)
            oov[wid] = all_unk

    return out, oov


def _has_leading_special(sentence_ids) -> bool:
    """True if position 0 of `sentence_ids` is a special token (e.g. RoBERTa-
    family <s>/BOS, which must stay at position 0)."""
    sent_list = sentence_ids.tolist()
    return bool(sent_list) and sent_list[0] in _tokenizer.all_special_ids


def _get_max_positions() -> Optional[int]:
    """Best-effort lookup of the model's maximum sequence length."""
    cfg = getattr(_model, "config", None)
    if cfg is None:
        return None
    for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
        val = getattr(cfg, attr, None)
        if val is not None:
            return int(val)
    return None


def _truncate_context(context_ids: list, sentence_len: int) -> list:
    """Drop the oldest (leftmost) context tokens so that
    `len(context_ids) + sentence_len` fits within the model's max length."""
    if not context_ids:
        return context_ids
    max_len = _get_max_positions()
    if max_len is None:
        return context_ids
    budget = max(max_len - sentence_len, 0)
    if len(context_ids) > budget:
        warnings.warn(
            f"context_text truncated from {len(context_ids)} to {budget} tokens "
            f"to fit within the model's max length ({max_len})."
        )
        context_ids = context_ids[len(context_ids) - budget:] if budget > 0 else []
    return context_ids


def _build_input_ids(context_ids: list, sentence_ids, has_leading_special: bool) -> torch.Tensor:
    """Concatenate context and sentence ids, keeping any leading special token
    at position 0 (RoBERTa-family models expect BOS/<s> at position 0)."""
    if not context_ids:
        return sentence_ids
    sent_list = sentence_ids.tolist()
    if has_leading_special:
        return torch.tensor(sent_list[:1] + context_ids + sent_list[1:], dtype=torch.long)
    return torch.tensor(context_ids + sent_list, dtype=torch.long)


def _sentence_subword_positions(n_sentence: int, context_len: int, has_leading_special: bool) -> List[int]:
    """Map sentence-relative subword positions (0..n_sentence-1, matching
    `subword_tokens`/`subword_word_ids`) to their indices in the full
    `input_ids` (= context + sentence) sequence, so that
    `subword_surprisals`/`subword_entropies` can be sliced to align with
    `subword_tokens`."""
    if context_len == 0:
        return list(range(n_sentence))
    if has_leading_special:
        return [0] + [context_len + i for i in range(1, n_sentence)]
    return [context_len + i for i in range(n_sentence)]


def score_masked_lm_tokens(
    tokens,
    temperature: float = 1.0,
    batch_size: int = 0,
    context_text: str = None,
    is_split_into_words: bool = True,
    pll_mode: str = "original",
    entropy: str = "onset",
    top_k: int = 0,
) -> Dict[str, List]:
    """Score a single sentence under an MLM.

    When `is_split_into_words=True` (default) `tokens` is a list of word
    strings and each element becomes one "word" in the per-word output.

    When `is_split_into_words=False` `tokens` is a raw sentence string and
    the LLM tokenizer's own pre-tokenizer decides where word boundaries fall;
    word indices come from `encoding.word_ids()`.

    `pll_mode`:
      - "original": mask one subword at a time (Salazar et al. 2020). Sibling
        subwords of a multi-subword word stay visible.
      - "within_word_l2r": when scoring subword k of a word, also mask all
        later subwords of that same word (Kauf & Ivanova 2023). Falls back to
        "original" when word_ids is unavailable.

    `entropy` controls how a multi-subword word's `word_entropies` value
    is derived from its subword entropies:
      - "onset" (default): entropy of the word's first subtoken.
      - "mean": mean across all subwords of the word.
      - "sum": sum across all subwords of the word.
      - "offset": entropy of the word's last subtoken.
      - "none": skip entropy entirely; `word_entropies` (and the subword
        entropies) are all NaN, and the full-vocab entropy sum is not computed.

    `top_k` > 0 additionally returns, for each scored subword position, the
    model's k most probable fill-ins for the mask at that position:
    `subword_topk_tokens` (list of k decoded strings per subword) and
    `subword_topk_probs` (their probabilities), aligned with
    `subword_tokens`. Positions that are never masked (special tokens) get
    empty lists.
    """
    global _tokenizer, _model, _device

    if _tokenizer is None or _model is None or _mode != "mlm":
        raise RuntimeError("Model not loaded in MLM mode. Call load_llm_model(mode='mlm') first.")

    if pll_mode not in ("original", "within_word_l2r"):
        raise ValueError("pll_mode must be one of: 'original', 'within_word_l2r'")

    if entropy not in _ENTROPY_MODES:
        raise ValueError(f"entropy must be one of: {_ENTROPY_MODES}")

    if is_split_into_words:
        if tokens is None or len(tokens) == 0:
            return _empty_result()
        tokens = ["" if t is None else str(t) for t in tokens]
    else:
        if tokens is None or not str(tokens).strip():
            return _empty_result()
        tokens = str(tokens)

    if context_text is not None:
        context_text = str(context_text)

    encoding = _tokenizer(
        tokens,
        is_split_into_words=is_split_into_words,
        return_tensors="pt",
        add_special_tokens=True,
    )
    sentence_ids = encoding["input_ids"][0]
    word_ids = _get_word_ids(encoding)

    context_ids = []
    if context_text is not None and str(context_text).strip():
        ctx = _tokenizer(context_text, add_special_tokens=False, return_tensors="pt")
        context_ids = ctx["input_ids"][0].tolist()

    has_leading_special = _has_leading_special(sentence_ids)
    context_ids = _truncate_context(context_ids, len(sentence_ids))
    input_ids = _build_input_ids(context_ids, sentence_ids, has_leading_special)
    context_len = len(context_ids)
    seq_len = input_ids.shape[0]
    sub_positions = _sentence_subword_positions(len(sentence_ids), context_len, has_leading_special)

    if is_split_into_words:
        n_words = len(tokens)
    else:
        valid_wids = [w for w in (word_ids or []) if w is not None]
        n_words = (max(valid_wids) + 1) if valid_wids else 0

    # Determine which positions to mask
    if word_ids is not None:
        positions = [context_len + i for i, wid in enumerate(word_ids) if wid is not None]
    else:
        special_ids = set()
        for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "bos_token_id", "eos_token_id"):
            tid = getattr(_tokenizer, attr, None)
            if tid is not None:
                special_ids.add(tid)
        positions = [
            context_len + i
            for i in range(len(sentence_ids))
            if int(sentence_ids[i].item()) not in special_ids
        ]

    # reticulate passes R numerics as Python float; coerce to int.
    # Named topk_n (not k) to avoid shadowing by the within_word_l2r loop's
    # `for k, target_pos in enumerate(group)` index below.
    topk_n = int(top_k) if top_k else 0

    if not positions:
        full_surprisals = [float("nan")] * seq_len
        full_entropies = [float("nan")] * seq_len
        result = {
            "subword_surprisals": [full_surprisals[p] for p in sub_positions],
            "subword_entropies": [full_entropies[p] for p in sub_positions],
            "subword_word_ids": word_ids if word_ids is not None else [],
            "subword_tokens": _tokenizer.convert_ids_to_tokens(sentence_ids.tolist()),
        }
        if topk_n > 0:
            result["subword_topk_tokens"] = [[] for _ in sub_positions]
            result["subword_topk_probs"] = [[] for _ in sub_positions]
        if word_ids is not None:
            result.update(_aggregate_to_words(word_ids, n_words, context_len,
                                              full_surprisals, full_entropies,
                                              entropy=entropy))
        else:
            result.update({"word_surprisals": [], "word_entropies": [], "word_token_counts": []})
        result["word_tokens"], result["word_oov"] = _extract_word_tokens(
            is_split_into_words, tokens, encoding, word_ids, sentence_ids, n_words
        )
        return result

    mask_id = _tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Model doesn't have a [MASK] token.")

    # Build (target_pos, extra_positions_to_also_mask) pairs.
    # For "within_word_l2r": when scoring subword k of a word, also mask later
    # subwords of that word so they can't leak the target's identity
    # (Kauf & Ivanova 2023). Falls back to "original" when word_ids unavailable.
    if pll_mode == "within_word_l2r" and word_ids is not None:
        word_groups: Dict[int, List[int]] = {}
        for sent_i, wid in enumerate(word_ids):
            if wid is None:
                continue
            word_groups.setdefault(wid, []).append(context_len + sent_i)
        variants: List = []
        for group in word_groups.values():
            for k, target_pos in enumerate(group):
                variants.append((target_pos, group[k + 1:]))
    else:
        variants = [(pos, []) for pos in positions]

    total = len(variants)
    # reticulate passes R numerics as Python float; coerce to int for range()
    chunk = int(batch_size) if batch_size and batch_size > 0 else total

    subword_surprisals = [float("nan")] * seq_len
    subword_entropies = [float("nan")] * seq_len
    subword_topk_tokens = [[] for _ in range(seq_len)]
    subword_topk_probs = [[] for _ in range(seq_len)]

    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        batch_variants = variants[start:end]

        masked_batch = []
        for target_pos, extra_positions in batch_variants:
            masked = input_ids.clone()
            masked[target_pos] = mask_id
            for extra in extra_positions:
                masked[extra] = mask_id
            masked_batch.append(masked)

        batch = torch.stack(masked_batch).to(_device)

        with torch.no_grad():
            outputs = _model(batch)
            logits_batch = outputs.logits  # (batch, seq_len, vocab)

        for local_idx, (target_pos, _extra) in enumerate(batch_variants):
            target_id = int(input_ids[target_pos].item())
            logits = logits_batch[local_idx, target_pos]
            surp, ent = _compute_surprisal_entropy(
                logits, target_id, temperature,
                compute_entropy=entropy != "none")
            subword_surprisals[target_pos] = surp
            subword_entropies[target_pos] = ent
            if topk_n > 0:
                toks, prbs = _topk_predictions(logits, topk_n, temperature)
                subword_topk_tokens[target_pos] = toks
                subword_topk_probs[target_pos] = prbs

    result = {
        "subword_surprisals": [subword_surprisals[p] for p in sub_positions],
        "subword_entropies": [subword_entropies[p] for p in sub_positions],
        "subword_word_ids": word_ids if word_ids is not None else [],
        "subword_tokens": _tokenizer.convert_ids_to_tokens(sentence_ids.tolist()),
    }
    if topk_n > 0:
        result["subword_topk_tokens"] = [subword_topk_tokens[p] for p in sub_positions]
        result["subword_topk_probs"] = [subword_topk_probs[p] for p in sub_positions]

    if word_ids is not None:
        result.update(_aggregate_to_words(word_ids, n_words, context_len,
                                          subword_surprisals, subword_entropies,
                                          entropy=entropy))
    else:
        result.update({"word_surprisals": [], "word_entropies": [], "word_token_counts": []})
    result["word_tokens"], result["word_oov"] = _extract_word_tokens(
        is_split_into_words, tokens, encoding, word_ids, sentence_ids, n_words
    )

    return result


def topk_whole_word_predictions(
    tokens,
    top_k: int = 5,
    temperature: float = 1.0,
    batch_size: int = 8,
    context_text: str = None,
    is_split_into_words: bool = True,
) -> Dict[str, List]:
    """Full-word alternatives for every word of a sentence under an MLM.

    Unlike ``score_masked_lm_tokens`` (which masks one subword at a time,
    leaving sibling pieces visible, so top-k at a piece position are word
    *completions*), this replaces ALL subword pieces of each word with a
    single [MASK] and reads the top-k of that distribution — the full words
    the model expected in the slot.

    Limitation: a single mask can only be filled by a single vocabulary
    token, so candidate words that the tokenizer would split into several
    pieces cannot appear among the predictions.

    Returns per word: `word_tokens`, `word_n_pieces`,
    `word_actual_prob` (probability of the word actually written when it is
    a single piece; NaN for multi-piece words, which live outside this
    distribution), `word_topk_tokens`, `word_topk_probs`.
    """
    global _tokenizer, _model, _device

    if _tokenizer is None or _model is None or _mode != "mlm":
        raise RuntimeError("Model not loaded in MLM mode. Call load_llm_model(mode='mlm') first.")

    if is_split_into_words:
        if tokens is None or len(tokens) == 0:
            return {"word_tokens": [], "word_n_pieces": [],
                    "word_actual_prob": [], "word_topk_tokens": [], "word_topk_probs": []}
        tokens = ["" if t is None else str(t) for t in tokens]
    else:
        if tokens is None or not str(tokens).strip():
            return {"word_tokens": [], "word_n_pieces": [],
                    "word_actual_prob": [], "word_topk_tokens": [], "word_topk_probs": []}
        tokens = str(tokens)

    mask_id = _tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Model doesn't have a [MASK] token.")

    encoding = _tokenizer(
        tokens,
        is_split_into_words=is_split_into_words,
        return_tensors="pt",
        add_special_tokens=True,
    )
    sentence_ids = encoding["input_ids"][0]
    word_ids = _get_word_ids(encoding)
    if word_ids is None:
        raise RuntimeError("Tokenizer does not expose word_ids(); cannot group subwords into words.")

    context_ids = []
    if context_text is not None and str(context_text).strip():
        ctx = _tokenizer(str(context_text), add_special_tokens=False, return_tensors="pt")
        context_ids = ctx["input_ids"][0].tolist()

    has_leading_special = _has_leading_special(sentence_ids)
    context_ids = _truncate_context(context_ids, len(sentence_ids))
    input_ids = _build_input_ids(context_ids, sentence_ids, has_leading_special)
    context_len = len(context_ids)
    pos_map = _sentence_subword_positions(len(sentence_ids), context_len, has_leading_special)

    if is_split_into_words:
        n_words = len(tokens)
    else:
        valid_wids = [w for w in (word_ids or []) if w is not None]
        n_words = (max(valid_wids) + 1) if valid_wids else 0

    # Word -> its (contiguous) positions in the full input_ids sequence.
    groups: Dict[int, List[int]] = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        groups.setdefault(wid, []).append(pos_map[i])

    # One variant per word: the word's whole span collapsed to a single mask.
    ids_list = input_ids.tolist()
    variants = []  # (wid, variant_ids, mask_pos)
    for wid in sorted(groups):
        span = groups[wid]
        variant = ids_list[:span[0]] + [mask_id] + ids_list[span[-1] + 1:]
        variants.append((wid, variant, span[0]))

    k = int(top_k)
    pad_id = _tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    topk_tokens: List[List[str]] = [[] for _ in range(n_words)]
    topk_probs: List[List[float]] = [[] for _ in range(n_words)]
    actual_prob = [float("nan")] * n_words

    chunk = int(batch_size) if batch_size and batch_size > 0 else max(len(variants), 1)
    for start in range(0, len(variants), chunk):
        batch_variants = variants[start:start + chunk]
        max_len = max(len(v[1]) for v in batch_variants)
        ids = torch.full((len(batch_variants), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch_variants), max_len), dtype=torch.long)
        for bi, (_, v_ids, _) in enumerate(batch_variants):
            ids[bi, :len(v_ids)] = torch.tensor(v_ids, dtype=torch.long)
            attn[bi, :len(v_ids)] = 1

        with torch.no_grad():
            out = _model(ids.to(_device), attention_mask=attn.to(_device))

        for bi, (wid, _v_ids, mask_pos) in enumerate(batch_variants):
            logits = out.logits[bi, mask_pos]
            toks, prbs = _topk_predictions(logits, k, temperature)
            topk_tokens[wid] = toks
            topk_probs[wid] = prbs
            span = groups[wid]
            if len(span) == 1:
                probs = torch.softmax(logits / temperature, dim=-1)
                actual_prob[wid] = float(probs[int(input_ids[span[0]].item())])

    word_tokens, _oov = _extract_word_tokens(
        is_split_into_words, tokens, encoding, word_ids, sentence_ids, n_words
    )

    return {
        "word_tokens": word_tokens,
        "word_n_pieces": [len(groups.get(w, [])) for w in range(n_words)],
        "word_actual_prob": actual_prob,
        "word_topk_tokens": topk_tokens,
        "word_topk_probs": topk_probs,
    }


_piece_masks_cache = None


def _piece_masks(vocab_dim: int):
    """Boolean masks over the logits dimension: which vocabulary items may
    start a word (`initial`) and which may only continue one (`continuation`).

    Detects the subword convention: WordPiece-style vocabularies mark
    continuations with a ## prefix; SentencePiece-style mark word starts with
    ▁ (or Ġ for byte-level BPE). Special tokens are excluded from both."""
    global _piece_masks_cache
    key = (vocab_dim, str(_device))
    if _piece_masks_cache is not None and _piece_masks_cache[0] == key:
        return _piece_masks_cache[1], _piece_masks_cache[2]

    vocab = _tokenizer.get_vocab()
    # Build as plain Python lists: torch CPU-tensor kernels can crash under
    # reticulate when R has loaded another OpenMP runtime (e.g. data.table),
    # so all tensor math stays on the model device.
    cont_l = [False] * vocab_dim
    init_l = [False] * vocab_dim
    has_wordpiece = any(t.startswith("##") for t in vocab)
    has_marker = any(t and t[0] in ("▁", "Ġ") for t in vocab)
    for tok, tid in vocab.items():
        if tid >= vocab_dim:
            continue
        if has_wordpiece:
            if tok.startswith("##"):
                cont_l[tid] = True
            else:
                init_l[tid] = True
        elif has_marker:
            if tok and tok[0] in ("▁", "Ġ"):
                init_l[tid] = True
            else:
                cont_l[tid] = True
        else:
            init_l[tid] = True
            cont_l[tid] = True
    for sid in _tokenizer.all_special_ids:
        if sid < vocab_dim:
            init_l[sid] = False
            cont_l[sid] = False
    init = torch.tensor(init_l, dtype=torch.bool, device=_device)
    cont = torch.tensor(cont_l, dtype=torch.bool, device=_device)
    _piece_masks_cache = (key, init, cont)
    return init, cont


def beam_word_predictions(
    tokens,
    top_k: int = 5,
    beam_width: int = 5,
    max_pieces: int = 3,
    temperature: float = 1.0,
    context_text: str = None,
    is_split_into_words: bool = True,
) -> Dict[str, List]:
    """Full-word alternatives for every word, including multi-piece words.

    For each word slot the span is replaced by n = 1..max_pieces [MASK]
    tokens; each n is filled left-to-right by beam search (continuation
    positions restricted to continuation pieces, word starts to word-start
    pieces), and candidates from all n are pooled, deduplicated on surface
    form, and ranked by joint probability (product of the chained piece
    probabilities under the model's full distribution — no renormalization,
    so single-piece probabilities match ``topk_whole_word_predictions``).

    Joint probabilities of different lengths are only heuristically
    comparable (longer sequences are penalized by the extra factors), which
    matches the intuition that a two-piece candidate must beat a one-piece
    candidate on both fills to outrank it.

    `word_actual_prob` is the written word's own joint probability under the
    same left-to-right chaining (defined for every word, unlike the
    single-mask version).

    Returns per word: `word_tokens`, `word_n_pieces`, `word_actual_prob`,
    `word_topk_tokens`, `word_topk_probs`, `word_topk_n_pieces`.
    """
    global _tokenizer, _model, _device

    if _tokenizer is None or _model is None or _mode != "mlm":
        raise RuntimeError("Model not loaded in MLM mode. Call load_llm_model(mode='mlm') first.")

    empty = {"word_tokens": [], "word_n_pieces": [], "word_actual_prob": [],
             "word_topk_tokens": [], "word_topk_probs": [], "word_topk_n_pieces": []}
    if is_split_into_words:
        if tokens is None or len(tokens) == 0:
            return empty
        tokens = ["" if t is None else str(t) for t in tokens]
    else:
        if tokens is None or not str(tokens).strip():
            return empty
        tokens = str(tokens)

    mask_id = _tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("Model doesn't have a [MASK] token.")

    encoding = _tokenizer(
        tokens,
        is_split_into_words=is_split_into_words,
        return_tensors="pt",
        add_special_tokens=True,
    )
    sentence_ids = encoding["input_ids"][0]
    word_ids = _get_word_ids(encoding)
    if word_ids is None:
        raise RuntimeError("Tokenizer does not expose word_ids(); cannot group subwords into words.")

    context_ids = []
    if context_text is not None and str(context_text).strip():
        ctx = _tokenizer(str(context_text), add_special_tokens=False, return_tensors="pt")
        context_ids = ctx["input_ids"][0].tolist()

    has_leading_special = _has_leading_special(sentence_ids)
    context_ids = _truncate_context(context_ids, len(sentence_ids))
    input_ids = _build_input_ids(context_ids, sentence_ids, has_leading_special)
    context_len = len(context_ids)
    pos_map = _sentence_subword_positions(len(sentence_ids), context_len, has_leading_special)

    if is_split_into_words:
        n_words = len(tokens)
    else:
        valid_wids = [w for w in (word_ids or []) if w is not None]
        n_words = (max(valid_wids) + 1) if valid_wids else 0

    groups: Dict[int, List[int]] = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        groups.setdefault(wid, []).append(pos_map[i])

    ids_list = input_ids.tolist()
    k = int(top_k)
    bw = int(beam_width)
    max_n = max(int(max_pieces), 1)
    ln_temp = float(temperature)

    def run_batch(variant_ids_batch):
        max_len = max(len(v) for v in variant_ids_batch)
        pad_id = _tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        ids = torch.full((len(variant_ids_batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(variant_ids_batch), max_len), dtype=torch.long)
        for bi, v in enumerate(variant_ids_batch):
            ids[bi, :len(v)] = torch.tensor(v, dtype=torch.long)
            attn[bi, :len(v)] = 1
        with torch.no_grad():
            out = _model(ids.to(_device), attention_mask=attn.to(_device))
        return out.logits

    def chained_log_probs(base_ids, p0, pieces):
        """Log-prob of each piece filled left-to-right (later ones masked)."""
        lps = []
        cur = list(base_ids)
        for i, pid in enumerate(pieces):
            logits = run_batch([cur])[0, p0 + i]
            lp = torch.log_softmax(logits / ln_temp, dim=-1)[pid].item()
            lps.append(lp)
            cur[p0 + i] = pid
        return lps

    topk_tokens: List[List[str]] = [[] for _ in range(n_words)]
    topk_probs: List[List[float]] = [[] for _ in range(n_words)]
    topk_n_pieces: List[List[int]] = [[] for _ in range(n_words)]
    actual_prob = [float("nan")] * n_words

    for wid in sorted(groups):
        span = groups[wid]
        prefix = ids_list[:span[0]]
        suffix = ids_list[span[-1] + 1:]
        p0 = span[0]

        # Written word's joint probability under the same chaining.
        actual_pieces = [ids_list[p] for p in span]
        base = prefix + [mask_id] * len(actual_pieces) + suffix
        actual_prob[wid] = math.exp(sum(chained_log_probs(base, p0, actual_pieces)))

        candidates = {}  # surface -> (joint_prob, n_pieces)
        for n in range(1, max_n + 1):
            base = prefix + [mask_id] * n + suffix
            beams = [([], 0.0)]  # (filled piece ids, cumulative log prob)
            for step in range(n):
                variant_batch = []
                for filled, _lp in beams:
                    cur = list(base)
                    for j, pid in enumerate(filled):
                        cur[p0 + j] = pid
                    variant_batch.append(cur)
                logits = run_batch(variant_batch)[:, p0 + step]
                log_probs = torch.log_softmax(logits / ln_temp, dim=-1)
                init_ok, cont_ok = _piece_masks(log_probs.shape[-1])
                allowed = cont_ok if step > 0 else init_ok
                # All tensor math on the model device (see _piece_masks note).
                log_probs = log_probs.masked_fill(~allowed, float("-inf"))
                new_beams = []
                for bi, (filled, lp) in enumerate(beams):
                    top = torch.topk(log_probs[bi], bw)
                    for tid, tlp in zip(top.indices.tolist(), top.values.tolist()):
                        if tlp == float("-inf"):
                            continue
                        new_beams.append((filled + [tid], lp + tlp))
                new_beams.sort(key=lambda b: b[1], reverse=True)
                beams = new_beams[:bw]
            for filled, lp in beams:
                pieces = _tokenizer.convert_ids_to_tokens(filled)
                try:
                    surface = _tokenizer.convert_tokens_to_string(pieces).strip()
                except Exception:
                    surface = "".join(pieces)
                prob = math.exp(lp)
                if surface not in candidates or prob > candidates[surface][0]:
                    candidates[surface] = (prob, n)

        ranked = sorted(candidates.items(), key=lambda c: c[1][0], reverse=True)[:k]
        topk_tokens[wid] = [c[0] for c in ranked]
        topk_probs[wid] = [c[1][0] for c in ranked]
        topk_n_pieces[wid] = [c[1][1] for c in ranked]

    word_tokens, _oov = _extract_word_tokens(
        is_split_into_words, tokens, encoding, word_ids, sentence_ids, n_words
    )

    return {
        "word_tokens": word_tokens,
        "word_n_pieces": [len(groups.get(w, [])) for w in range(n_words)],
        "word_actual_prob": actual_prob,
        "word_topk_tokens": topk_tokens,
        "word_topk_probs": topk_probs,
        "word_topk_n_pieces": topk_n_pieces,
    }


def score_autoregressive_tokens(
    tokens,
    temperature: float = 1.0,
    context_text: str = None,
    is_split_into_words: bool = True,
    entropy: str = "onset",
) -> Dict[str, List]:
    """AR counterpart of ``score_masked_lm_tokens``; see that function's
    docstring for the meaning of ``is_split_into_words`` and ``entropy``.
    """
    global _tokenizer, _model, _device, _mode

    if _tokenizer is None or _model is None or _mode != "ar":
        raise RuntimeError("Model not loaded in AR mode. Call load_llm_model(mode='ar') first.")

    if entropy not in _ENTROPY_MODES:
        raise ValueError(f"entropy must be one of: {_ENTROPY_MODES}")

    if is_split_into_words:
        if tokens is None or len(tokens) == 0:
            return _empty_result()
        tokens = ["" if t is None else str(t) for t in tokens]
    else:
        if tokens is None or not str(tokens).strip():
            return _empty_result()
        tokens = str(tokens)

    if context_text is not None:
        context_text = str(context_text)

    encoding = _tokenizer(
        tokens,
        is_split_into_words=is_split_into_words,
        return_tensors="pt",
        add_special_tokens=True,
    )
    sentence_ids = encoding["input_ids"][0]
    word_ids = _get_word_ids(encoding)

    if is_split_into_words:
        n_words = len(tokens)
    else:
        valid_wids = [w for w in (word_ids or []) if w is not None]
        n_words = (max(valid_wids) + 1) if valid_wids else 0

    context_ids = []
    if context_text is not None and str(context_text).strip():
        ctx = _tokenizer(context_text, add_special_tokens=False, return_tensors="pt")
        context_ids = ctx["input_ids"][0].tolist()

    has_leading_special = _has_leading_special(sentence_ids)
    context_ids = _truncate_context(context_ids, len(sentence_ids))
    input_ids = _build_input_ids(context_ids, sentence_ids, has_leading_special)
    context_len = len(context_ids)
    seq_len = input_ids.shape[0]
    sub_positions = _sentence_subword_positions(len(sentence_ids), context_len, has_leading_special)
    input_tensor = input_ids.unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = _model(input_tensor)
        logits = outputs.logits[0]  # (seq_len, vocab)

    subword_surprisals = [float("nan")] * seq_len
    subword_entropies = [float("nan")] * seq_len

    for pos in range(seq_len):
        if pos < context_len:
            continue
        if pos == 0:
            # Use BOS context if available; otherwise skip
            if getattr(_tokenizer, "bos_token_id", None) is None:
                continue
            bos_id = int(_tokenizer.bos_token_id)
            # Minimal logits from BOS-only input
            with torch.no_grad():
                bos_logits = _model(torch.tensor([[bos_id]], device=_device)).logits[0, -1]
            target_id = int(input_ids[pos].item())
            surp, ent = _compute_surprisal_entropy(
                bos_logits, target_id, temperature,
                compute_entropy=entropy != "none")
        else:
            target_id = int(input_ids[pos].item())
            surp, ent = _compute_surprisal_entropy(
                logits[pos - 1], target_id, temperature,
                compute_entropy=entropy != "none")

        subword_surprisals[pos] = surp
        subword_entropies[pos] = ent

    result = {
        "subword_surprisals": [subword_surprisals[p] for p in sub_positions],
        "subword_entropies": [subword_entropies[p] for p in sub_positions],
        "subword_word_ids": word_ids if word_ids is not None else [],
        "subword_tokens": _tokenizer.convert_ids_to_tokens(sentence_ids.tolist()),
    }

    if word_ids is not None:
        result.update(_aggregate_to_words(word_ids, n_words, context_len,
                                          subword_surprisals, subword_entropies,
                                          entropy=entropy))
    else:
        result.update({"word_surprisals": [], "word_entropies": [], "word_token_counts": []})
    result["word_tokens"], result["word_oov"] = _extract_word_tokens(
        is_split_into_words, tokens, encoding, word_ids, sentence_ids, n_words
    )

    return result
