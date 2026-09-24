"""Neural POS-sequence surprisal/entropy backend for ALSI.

Loads the trained UPOS next-tag language model (a small GRU exported as
`models/pos_lm_fr_gsd_alsi.pt`) and scores tag sequences with their FULL left
context, unlike the fixed-order trigram model in `R/fnt_pos_surprisal.R`:

    surprisal(t_i) = -log2 P(t_i | <bos>, t_1, ..., t_{i-1})
    entropy(t_i)   = H( P(. | <bos>, t_1, ..., t_{i-1}) )   in bits

The model is trained on the ALSI verb-modified tag scheme (copular 'etre' tagged
VERB rather than AUX), so a corpus parsed with the ALSI UDPipe model is scored
under the matching convention.

This module is self-contained (no dependency on the training repo) so that
`reticulate::source_python()` can load it directly. The model and vocabulary are
cached in the Python session and only reloaded when the checkpoint path changes,
mirroring the caching in `llm_scoring.py`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"

_STATE = {"model": None, "itos": None, "stoi": None, "device": None,
          "model_path": None}


class _UPOSLanguageModel(nn.Module):
    """Must match the architecture used at training time (see training repo
    src/model.py). Hyperparameters come from the checkpoint, so only the layer
    layout needs to agree."""

    def __init__(self, vocab_size, pad_id, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, lengths=None):
        emb = self.embed(inputs)
        if lengths is not None:
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = pad_packed_sequence(out, batch_first=True,
                                         total_length=inputs.size(1))
        else:
            out, _ = self.gru(emb)
        return self.proj(self.dropout(out))


def _pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_pos_lm_state():
    """Return the cached model metadata, or None if nothing is loaded."""
    if _STATE["model"] is None:
        return None
    return {"model_path": _STATE["model_path"],
            "vocab_size": len(_STATE["itos"]),
            "device": str(_STATE["device"])}


def load_pos_lm(model_path):
    """Load (and cache) the checkpoint at `model_path`."""
    if _STATE["model"] is not None and _STATE["model_path"] == model_path:
        return
    device = _pick_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = _UPOSLanguageModel(len(ckpt["itos"]), ckpt["pad_id"],
                               emb_dim=cfg["EMB_DIM"], hidden_dim=cfg["HIDDEN_DIM"],
                               num_layers=cfg["NUM_LAYERS"], dropout=cfg["DROPOUT"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    _STATE.update(model=model, itos=ckpt["itos"], stoi=ckpt["stoi"],
                  device=device, model_path=model_path)


def score_pos_sentences(sentences, model_path):
    """Score a list of tag sequences with full-context surprisal and entropy.

    Args:
        sentences: list of lists of UPOS tag strings (one inner list per
            sentence, tags in surface order).
        model_path: path to the .pt checkpoint.

    Returns:
        list (parallel to `sentences`) of dicts:
          {"surprisal": [float, ...], "entropy": [float, ...]}
        in bits, aligned position-by-position with the input tags. Unknown tags
        get NaN surprisal (and break the left context as <pad>).
    """
    load_pos_lm(model_path)
    model, itos, stoi, device = (_STATE["model"], _STATE["itos"],
                                 _STATE["stoi"], _STATE["device"])
    pad_id, bos_id = stoi[PAD], stoi[BOS]

    valid = torch.ones(len(itos), dtype=torch.bool, device=device)
    valid[pad_id] = False
    valid[bos_id] = False
    ln2 = 0.6931471805599453  # math.log(2)

    out = []
    with torch.no_grad():
        for tags in sentences:
            # reticulate marshals a length-1 R character vector to a Python
            # str; list(str) would explode it into characters. Guard first.
            if isinstance(tags, str):
                tags = [tags]
            tags = list(tags)
            if not tags:
                out.append({"surprisal": [], "entropy": []})
                continue
            ids = [stoi.get(t) for t in tags]
            enc = [bos_id] + [pad_id if i is None else i for i in ids]
            x = torch.tensor([enc[:-1]], dtype=torch.long, device=device)
            logits = model(x)[0].masked_fill(~valid, float("-inf"))
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            plogp = torch.where(p > 0, p * logp, torch.zeros_like(p))
            ent_bits = (-plogp.sum(dim=-1) / ln2).tolist()

            surprisal = []
            for pos, tid in enumerate(ids):
                if tid is None:
                    surprisal.append(float("nan"))
                else:
                    surprisal.append(-logp[pos, tid].item() / ln2)
            out.append({"surprisal": surprisal, "entropy": ent_bits})
    return out
