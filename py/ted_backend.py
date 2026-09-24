"""Tree Edit Distance backend for ALSI using the APTED algorithm.

Computes Tree Edit Distance (TED) between two dependency trees represented
as flat parallel lists (token_ids, head_ids, labels).

Loaded once via reticulate::source_python(); called as reticulate::py$compute_ted()
and reticulate::py$compute_ted_batch().

APTED is the current state-of-the-art exact TED algorithm (Pawlik & Augsten 2015),
O(n^3) worst-case but faster in practice than Zhang-Shasha for typical sentence trees.
"""

try:
    from apted import APTED, Config
except ImportError as e:
    raise ImportError(
        "apted not installed. Run: pip install apted"
    ) from e


class _Node:
    __slots__ = ("label", "children")

    def __init__(self, label):
        self.label = label
        self.children = []


class _AlsiConfig(Config):
    def rename(self, node1, node2):
        return 0 if node1.label == node2.label else 1

    def children(self, node):
        return node.children


def _ensure_list(x):
    """Coerce scalar or R-vector to Python list (reticulate edge case for length-1)."""
    if isinstance(x, str):
        return [x]
    return list(x)


def _build_tree(token_ids, head_ids, labels):
    """Build a rooted _Node tree from flat parallel lists.

    Tokens with head_id == "0" are attached to a virtual ROOT node.
    Tokens whose head is not found in token_ids (malformed trees) are
    also attached to ROOT to stay robust.
    """
    token_ids = _ensure_list(token_ids)
    head_ids  = _ensure_list(head_ids)
    labels    = _ensure_list(labels)

    nodes = {tid: _Node(lab) for tid, lab in zip(token_ids, labels)}
    root = _Node("ROOT")

    for tid, hid in zip(token_ids, head_ids):
        if hid == "0" or hid not in nodes:
            root.children.append(nodes[tid])
        else:
            nodes[hid].children.append(nodes[tid])

    return root, len(token_ids)


def compute_ted(token_ids_a, head_ids_a, labels_a,
                token_ids_b, head_ids_b, labels_b,
                normalize=True):
    """Compute Tree Edit Distance between two dependency trees.

    Parameters
    ----------
    token_ids_a, head_ids_a, labels_a : parallel sequences for tree A
    token_ids_b, head_ids_b, labels_b : parallel sequences for tree B
    normalize : bool
        If True, divide raw TED by max(size_a, size_b) + 1 (the +1
        accounts for the virtual ROOT node) so the result is in [0, 1].

    Returns
    -------
    float
        Raw TED (integer cast to float) if normalize=False, else
        normalized TED in [0, 1].
    """
    tree_a, n_a = _build_tree(token_ids_a, head_ids_a, labels_a)
    tree_b, n_b = _build_tree(token_ids_b, head_ids_b, labels_b)

    ted = APTED(tree_a, tree_b, _AlsiConfig()).compute_edit_distance()

    if normalize:
        denom = max(n_a, n_b) + 1  # +1 for ROOT
        return ted / denom if denom > 0 else 0.0

    return float(ted)


def compute_ted_batch(pairs, normalize=True):
    """Compute TED for a list of sentence pairs.

    Parameters
    ----------
    pairs : list of dict, each with keys:
        "token_ids_a", "head_ids_a", "labels_a",
        "token_ids_b", "head_ids_b", "labels_b"
    normalize : bool

    Returns
    -------
    list of float — one TED value per pair, same order as input.
    """
    results = []
    for p in pairs:
        results.append(compute_ted(
            p["token_ids_a"], p["head_ids_a"], p["labels_a"],
            p["token_ids_b"], p["head_ids_b"], p["labels_b"],
            normalize=normalize,
        ))
    return results
