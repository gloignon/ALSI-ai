"""Marginal word-level entropy for autoregressive (GPT-style) language models.

Motivation
----------
``llm_scoring.py`` already returns per-word *surprisal* (sum of sub-token
surprisals — exact, by the chain rule) and several per-word *entropy*
summaries (``entropy`` in {mean, sum, onset, offset, none}). Those entropy
summaries are heuristics built from the entropies of the *realised* sub-token
positions: none of them is the entropy of an actual distribution over whole
words.

The principled quantity for a causal LM is the entropy of the model's
predictive distribution over the *next word* given the left context:

    H[P(W | left)]  =  -Sum_w  P(w | left) * log P(w | left)

where ``w`` ranges over whole words and each ``P(w | left)`` is a product of
sub-token probabilities along the word's path. This cannot be recovered by any
aggregation of per-position sub-token entropies, because it requires
*marginalising* over every possible sub-token continuation that completes a
word. This module computes it by prefix-beam marginalisation.

Why only AR (and not MLM)
-------------------------
A masked LM has no proper autoregressive joint over a word's sub-tokens (its
per-slot scores are pseudo-log-likelihoods, not a factorised joint), so a
"marginal next-word distribution" is undefined. For MLMs the honest word-level
entropy remains the summed/onset per-slot entropy already provided by
``llm_scoring.score_masked_lm_tokens(entropy=...)``. This module therefore
covers the AR case only and is meant to *complement*, not replace, that file.

Word boundaries
---------------
A "word" here is a maximal run of sub-tokens beginning with a word-initial
sub-token (one carrying the tokenizer's space marker) and continuing with
non-word-initial sub-tokens. The word-initial predicate is detected from the
tokenizer family:

    * byte-level BPE (GPT-2 / RoBERTa / most FR AR models): starts with 'Ġ'
    * SentencePiece (CamemBERT / Flaubert / Llama):          starts with '▁'
    * WordPiece (BERT):                                      does NOT start '##'

This matches the segmentation that ``encoding.word_ids()`` produces in
``llm_scoring.py``, so the per-word entropies here align row-for-row with the
per-word surprisals computed there.

Model reuse
-----------
The functions reuse whatever AR model ``llm_scoring.load_llm_model(mode='ar')``
already loaded — no second copy of the weights. When called through
``reticulate::source_python`` the model globals live in ``__main__``; when used
from plain Python they are read from the imported ``llm_scoring`` module. You
may also pass ``tokenizer``/``model``/``device`` explicitly to override.
"""

import math
import sys
from typing import Dict, List, Optional, Sequence

import torch

LN2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# Model-state resolution (reuse the model loaded by llm_scoring.py)
# --------------------------------------------------------------------------- #
def _resolve_state(tokenizer=None, model=None, device=None, mode=None):
    """Return (tokenizer, model, device, mode), falling back to the AR model
    already loaded by ``llm_scoring`` (in ``__main__`` under reticulate, or in
    the imported module under plain Python)."""
    if tokenizer is not None and model is not None:
        return tokenizer, model, device, mode

    sources = []
    main = sys.modules.get("__main__")
    if main is not None:
        sources.append(main)
    ls = sys.modules.get("llm_scoring")
    if ls is None:
        try:
            import llm_scoring as ls  # type: ignore
        except Exception:
            ls = None
    if ls is not None:
        sources.append(ls)

    for src in sources:
        if getattr(src, "_model", None) is not None and getattr(src, "_tokenizer", None) is not None:
            return (
                getattr(src, "_tokenizer"),
                getattr(src, "_model"),
                getattr(src, "_device", None),
                getattr(src, "_mode", None),
            )

    raise RuntimeError(
        "No loaded model found. Call llm_scoring.load_llm_model(mode='ar') "
        "first, or pass tokenizer/model/device explicitly."
    )


# --------------------------------------------------------------------------- #
# Word-initial sub-token mask (cached per tokenizer)
# --------------------------------------------------------------------------- #
_WORD_INITIAL_CACHE: Dict[int, torch.Tensor] = {}


def _detect_family(sample_tokens: Sequence[str]) -> str:
    has_g = any(t.startswith("Ġ") for t in sample_tokens)
    has_meta = any(t.startswith("▁") for t in sample_tokens)
    has_hash = any(t.startswith("##") for t in sample_tokens)
    if has_g:
        return "bpe"
    if has_meta:
        return "spm"
    if has_hash:
        return "wordpiece"
    # Fallback: treat everything as word-initial (degenerate, but safe).
    return "unknown"


def _word_initial_mask(tokenizer) -> torch.Tensor:
    """Boolean tensor of shape (vocab,): True where the sub-token would begin a
    new word. Special tokens are always False (they never form part of a word).
    Cached per tokenizer instance."""
    key = id(tokenizer)
    cached = _WORD_INITIAL_CACHE.get(key)
    if cached is not None:
        return cached

    vocab_size = len(tokenizer)
    all_tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    # Sample a chunk to detect the marker family cheaply.
    family = _detect_family(all_tokens[: min(vocab_size, 5000)])
    special = set(getattr(tokenizer, "all_special_ids", []) or [])

    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i, tok in enumerate(all_tokens):
        if i in special or tok is None:
            continue
        if family == "bpe":
            is_init = tok.startswith("Ġ")
        elif family == "spm":
            is_init = tok.startswith("▁")
        elif family == "wordpiece":
            is_init = not tok.startswith("##")
        else:
            is_init = True
        mask[i] = is_init

    _WORD_INITIAL_CACHE[key] = mask
    return mask


# --------------------------------------------------------------------------- #
# Shared: entropy (bits) from a list of captured word/filling probabilities
# --------------------------------------------------------------------------- #
def _entropy_from_masses(masses: List[float]) -> Dict[str, float]:
    """Given the probabilities of the enumerated (beam-captured) words/fillings,
    return the entropy of the truncated distribution renormalised to the
    captured mass, plus diagnostics. ``entropy_bits_raw`` uses the unnormalised
    masses (a lower bound on the true entropy)."""
    captured = sum(masses)
    if captured <= 0:
        return {"entropy_bits": float("nan"), "captured_mass": 0.0,
                "entropy_bits_raw": float("nan")}
    raw = 0.0
    norm = 0.0
    for p in masses:
        if p > 0:
            raw -= p * math.log(p)
            q = p / captured
            norm -= q * math.log(q)
    return {
        "entropy_bits": norm / LN2,
        "captured_mass": captured,
        "entropy_bits_raw": raw / LN2,
    }


# --------------------------------------------------------------------------- #
# Core: marginal next-word entropy via prefix-beam marginalisation
# --------------------------------------------------------------------------- #
def ar_next_word_entropy(
    prefix_ids: Sequence[int],
    *,
    tokenizer=None,
    model=None,
    device=None,
    top_k: int = 40,
    max_word_len: int = 8,
    min_mass: float = 0.995,
    max_beams: int = 256,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Entropy (in bits) of the predictive distribution over the *next word*,
    given a left-context token sequence ``prefix_ids``.

    The next word is conditioned on starting at a word boundary: its first
    sub-token must be word-initial. A word ``w = (t1..tk)`` then has

        P(w) = [p0(t1) / Z0] * Prod_{j>=2} p_{j-1}(t_j) * Z_k

    where ``p_m`` is the next-token distribution after ``t1..tm``,
    ``Z_m = Sum_{v word-initial} p_m(v)`` is the probability the word terminates
    after ``m`` tokens, and ``Z0`` renormalises the first token onto the
    word-initial set. Summed over all words this is a proper distribution; the
    beam captures its high-probability head.

    Parameters
    ----------
    top_k          Branching factor when extending a partial word (per step).
    max_word_len   Hard cap on sub-tokens per word; longer paths are terminated.
    min_mass       Stop expanding once captured word mass reaches this fraction.
    max_beams      Cap on simultaneously active partial-word beams (pruned by
                   probability), bounding cost.
    temperature    Softmax temperature applied to every distribution.

    Returns
    -------
    dict with:
        entropy_bits     H over the captured words, renormalised to the
                         captured mass (so it is a valid entropy of the
                         truncated, renormalised distribution).
        captured_mass    Total probability of enumerated words (<= 1). Closer to
                         1.0 means a tighter approximation; report this.
        n_words          Number of distinct words enumerated.
        entropy_bits_raw H computed on the *unnormalised* captured masses
                         (lower bound on the true entropy; diagnostic only).
    """
    tokenizer, model, device, _ = _resolve_state(tokenizer, model, device)
    if device is None:
        device = next(model.parameters()).device
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    wi_mask = _word_initial_mask(tokenizer).to(device)
    prefix = torch.tensor(list(prefix_ids), dtype=torch.long, device=device)

    @torch.no_grad()
    def _next_probs(seqs: torch.Tensor) -> torch.Tensor:
        """seqs: (B, L) -> (B, vocab) probabilities for the next token."""
        logits = model(seqs).logits[:, -1, :] / temperature
        return torch.softmax(logits, dim=-1)

    # ---- Step 0: first token of the next word (conditioned word-initial) ----
    p0 = _next_probs(prefix.unsqueeze(0))[0]  # (vocab,)
    z0 = float(p0[wi_mask].sum().item())
    if z0 <= 0:
        return {"entropy_bits": float("nan"), "captured_mass": 0.0,
                "n_words": 0, "entropy_bits_raw": float("nan")}

    p0_wi = p0.clone()
    p0_wi[~wi_mask] = 0.0
    p0_wi = p0_wi / z0
    k = min(top_k, int((p0_wi > 0).sum().item()))
    top_p, top_i = torch.topk(p0_wi, k)

    # Active beams: list of (token_list, path_prob). path_prob already includes
    # the /Z0 conditioning and all continuation factors so far (NOT termination).
    beams: List[tuple] = [([int(i)], float(p)) for p, i in zip(top_p.tolist(), top_i.tolist())]

    completed_probs: List[float] = []  # P(word) for each enumerated word

    for _depth in range(max_word_len):
        if not beams:
            break
        # Prune to the most probable beams to bound cost.
        beams.sort(key=lambda b: b[1], reverse=True)
        beams = beams[:max_beams]

        # Batch one forward pass over all active beams.
        seqs = torch.stack([
            torch.cat([prefix, torch.tensor(toks, dtype=torch.long, device=device)])
            for toks, _ in beams
        ])
        probs = _next_probs(seqs)  # (B, vocab)

        z = probs[:, wi_mask].sum(dim=-1)  # (B,) termination prob per beam

        new_beams: List[tuple] = []
        for bi, (toks, path_p) in enumerate(beams):
            # Termination: the next token starts a new word -> this word is done.
            completed_probs.append(path_p * float(z[bi].item()))
            if len(toks) >= max_word_len:
                continue
            # Continuation: extend with non-word-initial tokens (top_k of them).
            cont = probs[bi].clone()
            cont[wi_mask] = 0.0
            kk = min(top_k, int((cont > 0).sum().item()))
            if kk == 0:
                continue
            cp, ci = torch.topk(cont, kk)
            for p, idx in zip(cp.tolist(), ci.tolist()):
                np_ = path_p * p
                if np_ > 0.0:
                    new_beams.append((toks + [int(idx)], np_))

        captured = sum(completed_probs)
        if captured >= min_mass:
            # Flush remaining beams as if they terminate now (upper-completes mass).
            for toks, path_p in new_beams:
                # approximate their eventual termination by current z is unknown;
                # attribute full remaining path prob as a single completed word.
                completed_probs.append(path_p)
            new_beams = []
        beams = new_beams

    # Any beams still active at max depth: count their path prob as completed.
    for toks, path_p in beams:
        completed_probs.append(path_p)

    res = _entropy_from_masses(completed_probs)
    res["n_words"] = len(completed_probs)
    return res


# --------------------------------------------------------------------------- #
# Sentence-level: predictive word entropy at each word boundary
# --------------------------------------------------------------------------- #
def score_ar_sentence_word_entropy(
    words: Sequence[str],
    *,
    context_text: Optional[str] = None,
    tokenizer=None,
    model=None,
    device=None,
    add_bos: bool = True,
    top_k: int = 40,
    max_word_len: int = 8,
    min_mass: float = 0.995,
    max_beams: int = 256,
    temperature: float = 1.0,
) -> Dict[str, List]:
    """Per-word marginal predictive entropy for one sentence under an AR model.

    For each word ``i`` the entropy is computed at the boundary *before* it,
    i.e. ``H[P(W | words[0..i-1] (+context))]`` — the model's uncertainty about
    the upcoming word given everything to its left. This is the forward-looking
    psycholinguistic entropy and aligns position-for-position with the per-word
    surprisals from ``llm_scoring.score_autoregressive_tokens``.

    Returns a dict of equal-length lists:
        word_entropies   entropy in bits at each word boundary
        captured_mass    beam-captured probability mass per word (report this)
        n_words_enum     distinct words enumerated per boundary
        word_tokens      the input words (echoed for alignment)
    """
    tokenizer, model, device, _ = _resolve_state(tokenizer, model, device)
    if device is None:
        device = next(model.parameters()).device

    words = ["" if w is None else str(w) for w in words]
    n = len(words)
    out_ent = [float("nan")] * n
    out_mass = [0.0] * n
    out_nw = [0] * n

    # Build the running prefix incrementally so each boundary reuses prior ids.
    prefix_ids: List[int] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    if add_bos and bos is not None:
        prefix_ids.append(int(bos))
    if context_text:
        ctx = tokenizer(str(context_text), add_special_tokens=False)["input_ids"]
        prefix_ids.extend(int(t) for t in ctx)

    enc = tokenizer(
        list(words), is_split_into_words=True, add_special_tokens=False
    )
    word_ids = enc.word_ids(0) if hasattr(enc, "word_ids") else None
    all_ids = enc["input_ids"]

    # Map each word -> its sub-token ids (in order), via word_ids alignment.
    per_word_ids: List[List[int]] = [[] for _ in range(n)]
    if word_ids is not None:
        for pos, wid in enumerate(word_ids):
            if wid is not None and 0 <= wid < n:
                per_word_ids[wid].append(int(all_ids[pos]))
    else:
        # Fallback: encode words one at a time.
        for i, w in enumerate(words):
            per_word_ids[i] = [int(t) for t in tokenizer(w, add_special_tokens=False)["input_ids"]]

    for i in range(n):
        if prefix_ids:  # need at least one context token to predict from
            res = ar_next_word_entropy(
                prefix_ids,
                tokenizer=tokenizer, model=model, device=device,
                top_k=top_k, max_word_len=max_word_len, min_mass=min_mass,
                max_beams=max_beams, temperature=temperature,
            )
            out_ent[i] = res["entropy_bits"]
            out_mass[i] = res["captured_mass"]
            out_nw[i] = res["n_words"]
        # Advance the prefix by this word's sub-tokens.
        prefix_ids = prefix_ids + per_word_ids[i]

    return {
        "word_entropies": out_ent,
        "captured_mass": out_mass,
        "n_words_enum": out_nw,
        "word_tokens": list(words),
    }


# --------------------------------------------------------------------------- #
# MLM: fixed-depth beam over a word's masked slots (within-word-l2r joint)
# --------------------------------------------------------------------------- #
def mlm_word_entropy_beam(
    input_ids: Sequence[int],
    target_positions: Sequence[int],
    *,
    tokenizer=None,
    model=None,
    device=None,
    top_k: int = 40,
    max_beams: int = 256,
    temperature: float = 1.0,
    restrict_word_shape: bool = False,
) -> Dict[str, float]:
    """Entropy (bits) of the within-word-l2r distribution over the fillings of a
    word that occupies the ``target_positions`` sub-token slots of ``input_ids``.

    The realised tokens at every *other* position form the fixed, bidirectional
    outer context. The word's own L = len(target_positions) slots are filled
    left-to-right by beam search: at slot j the model sees the outer context,
    the beam's chosen tokens for slots < j, and ``[MASK]`` at slots >= j — i.e.
    exactly the masking of ``pll_mode='within_word_l2r'`` in
    ``llm_scoring.score_masked_lm_tokens``. The resulting distribution

        Q(v_1..v_L) = Prod_j P(v_j | outer ctx, v_<j, slots>=j masked)

    is a proper joint (chain rule over a fixed-length span); its realised value
    has surprisal equal to the PLL-l2r word surprisal, so this entropy is the
    matching uncertainty measure. L = 1 reduces to the single masked-slot
    entropy.

    ``restrict_word_shape=True`` constrains slot 0 to word-initial sub-tokens
    and later slots to continuations (renormalising each), giving the entropy
    over *well-formed single words* of length L rather than over all length-L
    sub-token sequences. Default False keeps exact consistency with the
    (unrestricted) PLL-l2r surprisal.

    Returns the same keys as ``ar_next_word_entropy`` (``entropy_bits``,
    ``captured_mass``, ``n_words`` = distinct fillings enumerated,
    ``entropy_bits_raw``).
    """
    tokenizer, model, device, _ = _resolve_state(tokenizer, model, device)
    if device is None:
        device = next(model.parameters()).device
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("Tokenizer/model has no [MASK] token; MLM beam needs one.")

    positions = list(target_positions)
    L = len(positions)
    if L == 0:
        return {"entropy_bits": float("nan"), "captured_mass": 0.0,
                "n_words": 0, "entropy_bits_raw": float("nan")}

    base = torch.tensor(list(input_ids), dtype=torch.long, device=device)
    wi_mask = _word_initial_mask(tokenizer).to(device) if restrict_word_shape else None

    @torch.no_grad()
    def _slot_probs(seqs: torch.Tensor, pos: int) -> torch.Tensor:
        logits = model(seqs).logits[:, pos, :] / temperature
        return torch.softmax(logits, dim=-1)

    # Pre-mask all of the word's slots in the base sequence.
    masked_base = base.clone()
    for p in positions:
        masked_base[p] = mask_id

    # Beams: (filled_token_ids_for_slots_0..j-1, path_prob).
    beams: List[tuple] = [([], 1.0)]

    for j in range(L):
        pos = positions[j]
        # Build one sequence per beam: slots < j filled, slots >= j masked.
        seqs = []
        for toks, _ in beams:
            s = masked_base.clone()
            for t_idx in range(j):
                s[positions[t_idx]] = toks[t_idx]
            seqs.append(s)
        probs = _slot_probs(torch.stack(seqs), pos)  # (B, vocab)

        if wi_mask is not None:
            allow = wi_mask if j == 0 else ~wi_mask
            probs = probs.clone()
            probs[:, ~allow] = 0.0
            denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs = probs / denom

        new_beams: List[tuple] = []
        kk = min(top_k, probs.shape[-1])
        top_p, top_i = torch.topk(probs, kk, dim=-1)
        for bi, (toks, path_p) in enumerate(beams):
            for p, idx in zip(top_p[bi].tolist(), top_i[bi].tolist()):
                if p > 0.0:
                    new_beams.append((toks + [int(idx)], path_p * p))
        new_beams.sort(key=lambda b: b[1], reverse=True)
        beams = new_beams[:max_beams]

    res = _entropy_from_masses([p for _, p in beams])
    res["n_words"] = len(beams)
    return res


def score_mlm_sentence_word_entropy(
    words: Sequence[str],
    *,
    context_text: Optional[str] = None,
    tokenizer=None,
    model=None,
    device=None,
    top_k: int = 40,
    max_beams: int = 256,
    temperature: float = 1.0,
    restrict_word_shape: bool = False,
) -> Dict[str, List]:
    """Per-word within-word-l2r entropy for one sentence under an MLM.

    Each word's entropy is the entropy of the joint distribution over its own
    sub-token slots, conditioned on the rest of the sentence (bidirectional) via
    :func:`mlm_word_entropy_beam`. Aligns position-for-position with the per-word
    PLL surprisals from ``llm_scoring.score_masked_lm_tokens`` and is the entropy
    analogue of ``pll_mode='within_word_l2r'``.

    Returns equal-length lists: ``word_entropies``, ``captured_mass``,
    ``n_words_enum`` (distinct fillings enumerated per word), ``word_tokens``.
    """
    tokenizer, model, device, _ = _resolve_state(tokenizer, model, device)
    if device is None:
        device = next(model.parameters()).device

    words = ["" if w is None else str(w) for w in words]
    n = len(words)
    out_ent = [float("nan")] * n
    out_mass = [0.0] * n
    out_nw = [0] * n
    if n == 0:
        return {"word_entropies": [], "captured_mass": [], "n_words_enum": [],
                "word_tokens": []}

    enc = tokenizer(list(words), is_split_into_words=True, add_special_tokens=True)
    sentence_ids = list(enc["input_ids"])
    word_ids = enc.word_ids(0) if hasattr(enc, "word_ids") else None

    # Optionally prepend bidirectional context (kept after any leading special).
    leading_special = bool(sentence_ids) and sentence_ids[0] in set(
        getattr(tokenizer, "all_special_ids", []) or []
    )
    context_ids = []
    if context_text and str(context_text).strip():
        context_ids = [int(t) for t in tokenizer(str(context_text),
                       add_special_tokens=False)["input_ids"]]

    if context_ids and leading_special:
        input_ids = sentence_ids[:1] + context_ids + sentence_ids[1:]
        shift = lambda i: i if i == 0 else i + len(context_ids)
    elif context_ids:
        input_ids = context_ids + sentence_ids
        shift = lambda i: i + len(context_ids)
    else:
        input_ids = sentence_ids
        shift = lambda i: i

    # Map each word -> its (shifted) slot positions in input_ids.
    per_word_pos: List[List[int]] = [[] for _ in range(n)]
    if word_ids is not None:
        for sent_i, wid in enumerate(word_ids):
            if wid is not None and 0 <= wid < n:
                per_word_pos[wid].append(shift(sent_i))

    for i in range(n):
        if not per_word_pos[i]:
            continue
        res = mlm_word_entropy_beam(
            input_ids, per_word_pos[i],
            tokenizer=tokenizer, model=model, device=device,
            top_k=top_k, max_beams=max_beams, temperature=temperature,
            restrict_word_shape=restrict_word_shape,
        )
        out_ent[i] = res["entropy_bits"]
        out_mass[i] = res["captured_mass"]
        out_nw[i] = res["n_words"]

    return {
        "word_entropies": out_ent,
        "captured_mass": out_mass,
        "n_words_enum": out_nw,
        "word_tokens": list(words),
    }


def score_mlm_raw_sentence_word_entropy(
    sentence: str,
    *,
    tokenizer=None,
    model=None,
    device=None,
    top_k: int = 40,
    max_beams: int = 256,
    temperature: float = 1.0,
    restrict_word_shape: bool = False,
) -> Dict[str, List]:
    """Raw-string counterpart of :func:`score_mlm_sentence_word_entropy`.

    The sentence is passed to the tokenizer as a raw string so the model's own
    pre-tokenizer decides word boundaries (``encoding.word_ids()``), matching
    the ``is_split_into_words=False`` path of
    ``llm_scoring.score_masked_lm_tokens``. This makes the per-word beam entropy
    align row-for-row with the per-word PLL surprisal cached by the L2-writing
    pipeline (which scores raw sentences).

    Returns equal-length lists keyed by word index:
    ``word_entropies``, ``captured_mass``, ``n_words_enum``, ``word_tokens``.
    """
    tokenizer, model, device, _ = _resolve_state(tokenizer, model, device)
    if device is None:
        device = next(model.parameters()).device

    sentence = "" if sentence is None else str(sentence)
    if not sentence.strip():
        return {"word_entropies": [], "captured_mass": [],
                "n_words_enum": [], "word_tokens": []}

    enc = tokenizer(sentence, add_special_tokens=True)
    input_ids = list(enc["input_ids"])
    word_ids = enc.word_ids(0) if hasattr(enc, "word_ids") else None
    if word_ids is None:
        raise RuntimeError(
            "Tokenizer did not return word_ids; raw-sentence beam needs a fast "
            "tokenizer. Use score_mlm_sentence_word_entropy with a word list."
        )

    valid = [w for w in word_ids if w is not None]
    n = (max(valid) + 1) if valid else 0
    if n == 0:
        return {"word_entropies": [], "captured_mass": [],
                "n_words_enum": [], "word_tokens": []}

    per_word_pos: List[List[int]] = [[] for _ in range(n)]
    for pos, wid in enumerate(word_ids):
        if wid is not None and 0 <= wid < n:
            per_word_pos[wid].append(pos)

    subword_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    out_ent = [float("nan")] * n
    out_mass = [0.0] * n
    out_nw = [0] * n
    word_tokens = [""] * n
    for i in range(n):
        positions = per_word_pos[i]
        if not positions:
            continue
        try:
            word_tokens[i] = tokenizer.convert_tokens_to_string(
                [subword_tokens[p] for p in positions]
            ).strip()
        except Exception:
            word_tokens[i] = "".join(subword_tokens[p] for p in positions)
        res = mlm_word_entropy_beam(
            input_ids, positions,
            tokenizer=tokenizer, model=model, device=device,
            top_k=top_k, max_beams=max_beams, temperature=temperature,
            restrict_word_shape=restrict_word_shape,
        )
        out_ent[i] = res["entropy_bits"]
        out_mass[i] = res["captured_mass"]
        out_nw[i] = res["n_words"]

    return {
        "word_entropies": out_ent,
        "captured_mass": out_mass,
        "n_words_enum": out_nw,
        "word_tokens": word_tokens,
    }


if __name__ == "__main__":
    # Smoke test: AR (open-ended beam) and MLM (fixed-depth within-word beam).
    import llm_scoring

    print("== AR (distilgpt2) ==")
    llm_scoring.load_llm_model("distilgpt2", mode="ar")
    r = score_ar_sentence_word_entropy(
        ["The", "cat", "sat", "on", "the", "mat"], top_k=40, max_beams=128
    )
    for w, e, m in zip(r["word_tokens"], r["word_entropies"], r["captured_mass"]):
        print(f"{w:>8}  H={e:8.3f} bits   captured={m:.3f}")

    print("== MLM (distilbert-base-uncased) ==")
    llm_scoring.load_llm_model("distilbert-base-uncased", mode="mlm")
    r = score_mlm_sentence_word_entropy(
        ["the", "antidisestablishmentarianism", "prevailed"], top_k=40, max_beams=128
    )
    for w, e, m, nw in zip(r["word_tokens"], r["word_entropies"],
                           r["captured_mass"], r["n_words_enum"]):
        print(f"{w:>28}  H={e:8.3f} bits   captured={m:.3f}  fillings={nw}")
