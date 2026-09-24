#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segment a TSV corpus (doc_id, text) into sentences with syntok.

Usage:
  python segment_sentences.py --in corpus.tsv --out sentences.tsv [--merge-colon-semicolon]
  # Optional knobs for safe merging:
  #   --max-prev-len 35 --max-next-len 35 --max-merged-len 60

Input  (TSV):  doc_id<TAB>text
Output (TSV):  doc_id<TAB>sentence_id<TAB>sentence
  - sentence_id restarts at 1 for each document
  - spacing preserved
"""

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    import syntok.segmenter as segmenter
except ImportError as e:
    sys.stderr.write("syntok not installed. Run: pip install syntok\n")
    raise

# --- Heuristic helpers for safe merging across ';' and ':' ---

# Words = letters (incl. accents) + digits, may contain internal hyphens (arc-en-ciel = 1).
# Apostrophes are NOT internal connectors, so l’avion = 2 words.
WORD_RE = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ]+(?:-[\wÀ-ÖØ-öø-ÿ]+)*\b", re.UNICODE)

# Common French “continuation” starters (feel free to extend)
CONTINUATION_WORDS = {
    "et", "mais", "ou", "donc", "or", "ni", "car",             # coord. conj.
    "cependant", "pourtant", "toutefois", "ainsi", "alors",
    "puis", "ensuite", "de", "du", "des", "par", "en", "avec",
    "autrement", "sinon", "néanmoins", "dès", "depuis"
}

def token_count(s: str) -> int:
    return len(WORD_RE.findall(s or ""))

def starts_like_continuation(s: str) -> bool:
    s2 = (s or "").lstrip()
    if not s2:
        return False
    # Lowercase first char is a cheap continuation cue (French sentences usually cap after a real stop)
    if s2[0].islower():
        return True
    m = WORD_RE.search(s2)
    if not m:
        return False
    return m.group(0).lower() in CONTINUATION_WORDS

def clause_heavy(s: str, max_commas: int = 3, max_em_dashes: int = 2) -> bool:
    s = s or ""
    return (s.count(",") >= max_commas) or (s.count("—") >= max_em_dashes)

def iter_rows(tsv_path):
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Require expected columns
        required = {"doc_id", "text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input must have columns: {required}; got {reader.fieldnames}")
        for row in reader:
            yield row["doc_id"], row["text"]

# Sentence-ending punctuation ([.!?]) preceded by a non-alphabetic character
# (digit, %, currency symbol, closing bracket/quote, etc.) that syntok may
# not split on.  We insert a newline so syntok treats it as a hard boundary.
# The lookbehind matches one non-alpha, non-whitespace character; the
# lookahead requires whitespace + uppercase to avoid firing inside e.g.
# "fig. 3" or "v2.0".
_NONSPLIT_END_RE = re.compile(
    r"(?<=[^\w\s,;:\-])"              # preceded by punctuation/symbol (%, ), ], », …)
    r"([.!?])"                         # sentence-ending mark
    r"(?=\s+[A-ZÀ-ÖØ-ÝŸ])",          # followed by whitespace + uppercase letter
    re.UNICODE,
)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return s.replace("\r\n", "\n").replace("\r", "\n")

def _insert_split_boundaries(text: str) -> str:
    """
    Insert a newline after sentence-ending punctuation that follows a
    non-alphabetic character, so syntok treats the boundary as a hard split.
    """
    return _NONSPLIT_END_RE.sub(r"\1\n", text)

def syntok_sentence_split(text: str) -> list[str]:
    """
    Use syntok for sentence segmentation - more robust than regex.
    """
    if not text:
        return []
    text = _insert_split_boundaries(text)
    sentences: list[str] = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            sentence_text = "".join(map(str, sentence)).strip()
            if sentence_text:
                sentences.append(sentence_text)
    return sentences

def should_merge(prev: str,
                 nxt: str,
                 max_prev_len: int,
                 max_next_len: int,
                 max_merged_len: int) -> bool:
    """
    Decide whether to merge prev (ending with ; or :) with nxt.
    Conditions:
      1) prev_len <= max_prev_len and next_len <= max_next_len
      2) prev_len + next_len <= max_merged_len
      3) nxt looks like a continuation (lowercase start or continuation word)
      4) neither side is clause-heavy
    """
    if not prev or not nxt:
        return False

    prev_len = token_count(prev)
    next_len = token_count(nxt)

    if prev_len > max_prev_len or next_len > max_next_len:
        return False
    if prev_len + next_len > max_merged_len:
        return False
    if clause_heavy(prev) or clause_heavy(nxt):
        return False
    if not starts_like_continuation(nxt):
        return False

    return True

def merge_colon_semicolon_sentences(sentences,
                                    max_prev_len: int = 35,
                                    max_next_len: int = 35,
                                    max_merged_len: int = 60):
    """
    Merge sentences that end with ':' or ';' with the following sentence,
    but only when the heuristic says it's safe.
    """
    if not sentences:
        return sentences

    merged = []
    i = 0
    while i < len(sentences):
        cur = sentences[i].strip()
        if i < len(sentences) - 1 and re.search(r"[;:]$", cur):
            nxt = sentences[i + 1].strip()
            if should_merge(cur, nxt, max_prev_len, max_next_len, max_merged_len):
                merged.append(f"{cur} {nxt}")
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged

def merge_short_sentences(sentences: list[str], min_sent_len: int = 4) -> list[str]:
    """
    Scan from the end: any sentence shorter than *min_sent_len* tokens is
    merged into the preceding sentence.  Scanning backward means a merged
    sentence that is still short gets merged again on the next iteration.
    If the first sentence is short and has no predecessor, it is kept as-is.
    """
    if min_sent_len <= 0 or len(sentences) <= 1:
        return sentences
    result = list(sentences)
    i = len(result) - 1
    while i > 0:
        if token_count(result[i]) < min_sent_len:
            result[i - 1] = f"{result[i - 1]} {result[i]}"
            del result[i]
        i -= 1
    return result


def segment_text_syntok(text: str,
                        min_sent_len: int = 4,
                        merge_colon_semicolon: bool = False,
                        max_prev_len: int = 35,
                        max_next_len: int = 35,
                        max_merged_len: int = 60) -> list[str]:
    """
    Segment *text* into sentences with syntok, then apply optional post-processing.

    Called directly by the R wrapper via reticulate::source_python().

    Parameters
    ----------
    text               : raw text string
    min_sent_len       : merge trailing sentences shorter than this many tokens
                         into the preceding sentence (0 = disabled)
    merge_colon_semicolon : apply colon/semicolon heuristic merge
    max_prev_len / max_next_len / max_merged_len : passed to
        merge_colon_semicolon_sentences when enabled
    """
    sents = syntok_sentence_split(normalize_text(text))
    if merge_colon_semicolon:
        sents = merge_colon_semicolon_sentences(
            sents,
            max_prev_len=max_prev_len,
            max_next_len=max_next_len,
            max_merged_len=max_merged_len,
        )
    sents = merge_short_sentences(sents, min_sent_len=min_sent_len)
    return [s.strip() for s in sents]


def main(args):
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    # Prepare writer with header
    with open(out_path, "w", encoding="utf-8", newline="") as fo:
        writer = csv.writer(fo, delimiter="\t", lineterminator="\n")
        writer.writerow(["doc_id", "sentence_id", "sentence"])

        for doc_id, text in iter_rows(in_path):
            text = normalize_text(text)
            # Segment into sentences
            sents = syntok_sentence_split(text) if text else []

            # Apply colon/semicolon merging if requested
            if args.merge_colon_semicolon:
                sents = merge_colon_semicolon_sentences(
                    sents,
                    max_prev_len=args.max_prev_len,
                    max_next_len=args.max_next_len,
                    max_merged_len=args.max_merged_len
                )

            # Merge short trailing sentences
            sents = merge_short_sentences(sents, min_sent_len=args.min_sent_len)

            # Number sentences starting at 1 for each doc
            for i, sent in enumerate(sents, start=1):
                writer.writerow([doc_id, i, sent.strip()])

if __name__ == "__main__" and len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="Sentence segmentation with syntok.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input TSV with columns: doc_id, text")
    parser.add_argument("--out", dest="out_path", required=True, help="Output TSV path")
    parser.add_argument("--merge-colon-semicolon", action="store_true",
                        help="Merge sentences ending with ':' or ';' with the following sentence (safely).")
    parser.add_argument("--max-prev-len", type=int, default=35,
                        help="Max tokens allowed for the sentence ending with ';' or ':' to be eligible for merge.")
    parser.add_argument("--max-next-len", type=int, default=35,
                        help="Max tokens allowed for the following sentence to be eligible for merge.")
    parser.add_argument("--max-merged-len", type=int, default=60,
                        help="Max tokens allowed for the merged sentence.")
    args = parser.parse_args()
    main(args)
