# ALSI Demo: Parsing Backends — UDPipe, spaCy, Trankit
#
# parse_text() supports three NLP backends that all return the same
# data.table format (one row per token, same columns). This demo shows
# how to switch backends and what to expect from each one.
#
# Backends at a glance:
#   "udpipe"  — default; fast, pure R, no extra setup; rule-assisted lemma
#               via the Lefff dictionary; ~96.7% UPOS, 97.4% lemma accuracy
#   "spacy"   — CNN-based; best sentence segmentation (100%); strong UPOS
#               (~97.1%); trainable lemmatizer (no Lefff, ~94.1% lemma)
#   "trankit" — XLM-RoBERTa transformer; highest UPOS (~98.3%) and dependency
#               accuracy (UAS 94.7%, LAS 92.2%); slower; requires ~1.1 GB
#               XLM-RoBERTa download on first use (cached afterwards)
#
# All three backends use the same retagged training data (copular être → VERB).
# All metrics are end-to-end on the standard UD French-GSD test set.
#
# Prerequisites:
#   - models/french_gsd-remix_3.udpipe    — UDPipe model
#   - models/spacy_fr_gsd_alsi_v1/        — spaCy model
#   - models/trankit_fr_v1/               — Trankit model
#   - demo_corpora/viki_wiki.zip          — optional full corpus run
#   (spaCy and Trankit Python packages are installed automatically on first use)
#
# Last update: 2026-05-27
#
library(data.table)
library(udpipe)

source("../alsi/R/fnt_corpus.R", encoding = "UTF-8")
source("../alsi/R/fnt_setup.R", encoding = "UTF-8")


# 1) Setup ----

# Model paths — edit these if your models live elsewhere.
udpipe_model_path  <- "models/french_gsd-remix_3.udpipe"
spacy_model_path   <- "models/spacy_fr_gsd_alsi_v1"
trankit_model_path <- "models/trankit_fr_v1"

# Load the UDPipe model into memory (needed only for the udpipe backend).
# spaCy and Trankit load lazily on first parse_text() call.
if (!file.exists(udpipe_model_path)) {
  message("French UDPipe model not found — downloading from GitHub (one-time, ~50 MB)...")
  source("../alsi/R/artefact_builders/fetch_udpipe_models.R")
}
udmodel_french <- udpipe_load_model(file = udpipe_model_path)

# spaCy and Trankit models are also too large for regular git history and are
# fetched from GitHub on first use if missing.
if (!dir.exists(spacy_model_path)) {
  message("French spaCy model not found — downloading from GitHub (one-time, ~8 MB)...")
  source("R/artefact_builders/fetch_spacy_models.R")
}
if (!dir.exists(file.path(trankit_model_path, "xlm-roberta-base", "customized-mwt"))) {
  message("French Trankit model not found — downloading from GitHub (one-time, ~34 MB)...")
  source("R/artefact_builders/fetch_trankit_models.R")
}

# Three sentences that exercise different linguistic features:
#   - copular être ("est heureuse") — our custom VERB tagging for the copula
#   - a relative clause ("que les enfants lisent")
#   -  a subordinate clause + past participle agreement ("inventées" agrees with "statistiques")
example_texts <- c(
  sent1 = "La fille est heureuse.",
  sent2 = "Ce livre que les enfants lisent est facile.",
  sent3 = "Je crois que 80% des statistiques sont inventées."
)


# 2) UDPipe backend (default) ----
#
# Fast, no extra setup. Lemmatisation is helped by the Lefff lexicon.
# Copular être is tagged VERB (our convention) with dep_rel = "cop".

dt_udpipe <- parse_text(example_texts, backend = "udpipe")

print(dt_udpipe[, .(doc_id, token_id, token, lemma, upos, dep_rel, head_token_id)])


# 3) spaCy backend ----
#
# CNN-based model trained on the same retagged French-GSD data. Fastest of
# the three on CPU after the first call. Produces the best sentence boundaries
# (100% on GSD test set) and strong UPOS (~97.1%), but lemmatisation relies
# on spaCy's lookup tables rather than the Lefff lexicon (~94.1% lemma accuracy).
#
# The model must be in models/spacy_fr_gsd_alsi_v1/ (default path).
# Python packages are installed automatically on first use — no manual setup.
# spaCy does not produce MWT rows, so token_id is always a plain integer.

dt_spacy <- parse_text(example_texts, backend = "spacy", spacy_model = spacy_model_path)

print(dt_spacy[, .(doc_id, token_id, token, lemma, upos, dep_rel, head_token_id)])


# 4) Trankit backend ----
#
# XLM-RoBERTa transformer model fine-tuned on the same retagged French-GSD data.
# Highest UPOS (~98.3%) and dependency accuracy (UAS 94.7%, LAS 92.2%); slower
# than the other two. Python packages and the XLM-RoBERTa base model (~1.1 GB)
# are downloaded from HuggingFace on the very first use and cached afterwards.
#
# The model files must be in models/trankit_fr_v1/ (same convention as UDPipe).

dt_trankit <- parse_text(example_texts, backend = "trankit", trankit_model = trankit_model_path)

print(dt_trankit[, .(doc_id, token_id, token, lemma, upos, dep_rel, head_token_id)])


# 5) Compare all three backends side by side ----
#
# Merge on doc_id + sentence_id + token_id to see where backends agree or differ.
# Differences are most common on lemmas of irregular verbs and on dependency
# relations in complex embedded clauses.
#
# UDPipe uses "6-7"-style token_ids for multi-word tokens (e.g. "des" = "de"+"les").
# Drop those span rows before joining — the individual sub-tokens follow immediately.
# spaCy and Trankit never produce MWT rows.

dt_udpipe  <- dt_udpipe |> filter(!grepl("-", token_id, fixed = TRUE))
dt_udpipe  <- dt_udpipe |> mutate(token_id = as.integer(token_id))
dt_trankit <- dt_trankit |> mutate(token_id = as.integer(token_id))
# dt_spacy token_id is already integer, no need to convert.

dt_compare <- merge(
  dt_udpipe [, .(doc_id, sentence_id, token_id, token,
                 upos_udpipe   = upos,
                 lemma_udpipe  = lemma,
                 dep_udpipe    = dep_rel)],
  dt_spacy  [, .(doc_id, sentence_id, token_id,
                 upos_spacy    = upos,
                 lemma_spacy   = lemma,
                 dep_spacy     = dep_rel)],
  by = c("doc_id", "sentence_id", "token_id")
)
dt_compare <- merge(
  dt_compare,
  dt_trankit[, .(doc_id, sentence_id, token_id,
                 upos_trankit  = upos,
                 lemma_trankit = lemma,
                 dep_trankit   = dep_rel)],
  by = c("doc_id", "sentence_id", "token_id")
)

# Flag tokens where any two backends disagree on UPOS or lemma.
dt_compare <- dt_compare |> mutate(
  upos_agree  = (upos_udpipe  == upos_spacy)  & (upos_spacy  == upos_trankit),
  lemma_agree = (lemma_udpipe == lemma_spacy) & (lemma_spacy == lemma_trankit)
)

print(dt_compare[, .(doc_id, token_id, token,
                     upos_udpipe, upos_spacy, upos_trankit, upos_agree,
                     lemma_udpipe, lemma_spacy, lemma_trankit, lemma_agree)])


# 6) Full corpus with spaCy ----
#
# For a whole corpus, the API is identical to UDPipe — just add backend = "spacy".
# spaCy is typically faster than Trankit on CPU for large batches.

if (file.exists("demo_corpora/viki_wiki.zip") || dir.exists("demo_corpora/viki_wiki")) {
  dt_txt <- load_demo_corpus()
  message(nrow(dt_txt), " documents loaded — parsing with spaCy...")
  dt_spacy_corpus <- parse_text(dt_txt, backend = "spacy", spacy_model = spacy_model_path)
} else {
  message("Place the Viki-Wiki corpus zip at demo_corpora/viki_wiki.zip for a full corpus run.")
}
