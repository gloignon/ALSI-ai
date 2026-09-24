# Demo: POS-trigram and neural POS-LM surprisal/entropy
#
# This demo shows how to measure how *predictable* the grammatical structure
# of a sentence is, using two backends trained on French UD data:
#
# - pos_surprisal(): a fixed-order POS trigram model. Fast, simple, but only
#   sees the two preceding tags.
# - pos_surprisal_nn(): a small GRU language model over UPOS sequences,
#   conditioned on the *entire* left context of the sentence. Slower (loads
#   PyTorch via reticulate) but scores every token, including the first one.
#
# POS surprisal (-log2 p) tells us how unexpected a word's grammatical
# category is given its left context. High surprisal = unusual local POS
# sequence; low surprisal = very predictable structure.
#
# POS entropy is the uncertainty over what category comes next given the
# same context. It is a property of the context, not the target token:
# high entropy means many continuations are possible from that position.
#
# Entropy reduction (entropy[t-1] - entropy[t]) captures how much the
# current token resolved the uncertainty created at the previous step.
#
# The trigram model is a raw frequency table — no smoothing yet. Unseen
# trigrams receive NA surprisal. With use_sentence_boundaries = FALSE
# (default), tokens at sentence positions 1 and 2 (no full left context) are
# unscored. The neural model has no such gap: a <bos> marker lets it score
# every position.
#
# Prerequisites:
# - out/demo_parsed_tagged.Rds — run demos/demo_parse_tag.R first (section 5).
# - models/french_gsd-remix_3.udpipe — the standard ALSI UDPipe model (same
#   verb-retagging convention as the POS trigram model); downloaded
#   automatically from the GitHub release if missing.
# - models/pos_lm_fr_gsd_alsi.pt — the neural POS-LM checkpoint, ships with
#   ALSI. Scoring requires reticulate + torch (installed automatically via
#   reticulate::py_require() on first use).

# 1) Setup ----

library(tidyverse)
library(udpipe)
library(data.table)

source("../alsi/R/fnt_pos_surprisal.R",    encoding = "UTF-8")
source("R/fnt_pos_surprisal_nn.R", encoding = "UTF-8")
source("../alsi/R/fnt_utility.R",          encoding = "UTF-8")

# Settings used when the distributed model was built. The same values must be
# passed to every pos_surprisal() / pos_surprisal_nn() call so scoring matches
# the models.
EXCLUDE_POS    <- c("PUNCT", "SYM")
USE_BOUNDARIES <- TRUE
POS_LM_PATH    <- "models/pos_lm_fr_gsd_alsi.pt"

# 2) Load the POS trigram model ----

# Trained on French-GSD with the ALSI verb-retagging convention, PUNCT/SYM
# excluded, sentence boundaries enabled. Ships with ALSI.
message("Loading distributed POS model...")
pos_model <- readRDS("models/pos_trigram_fr_gsd_alsi.Rds")


# 3) Score individual sentences ----

alsi_udpipe_path <- "models/french_gsd-remix_3.udpipe"
if (!file.exists(alsi_udpipe_path)) {
  message("ALSI UDPipe model not found — downloading from GitHub (one-time, ~68 MB)...")
  source("../alsi/R/artefact_builders/fetch_udpipe_models.R")
}
udmodel <- udpipe_load_model(alsi_udpipe_path)

parse_one <- function(text) {
  as_tibble(udpipe_annotate(udmodel, x = text))
}

# ── 3a. A syntactically predictable sentence ─────────────────────────────────
#
# "Le petit chat mange la souris."
# DET ADJ NOUN VERB DET NOUN — a canonical SVO sequence; every transition
# is common in French. Expect uniformly low surprisal.

message("--- Predictable: 'Le petit chat mange la souris.' ---")
dt_easy <- parse_one("Le petit chat mange la souris.")
res_easy <- pos_surprisal(dt_easy, pos_model, exclude_pos = EXCLUDE_POS,
                          use_sentence_boundaries = USE_BOUNDARIES)

print(res_easy$token_surprisal |>
  as_tibble() |>
  left_join(as_tibble(dt_easy)[, c("token_id", "token")], by = "token_id") |>
  select(token, upos, pos_surprisal, pos_entropy, pos_entropy_reduction) |>
  mutate(across(where(is.numeric), \(x) round(x, 2))))

# ── 3b. A syntactically surprising sentence ───────────────────────────────────
#
# "Hier soir, étrangement, il pleuvait des cordes."
# ADV NOUN ADV PRON VERB DET NOUN — the opening ADV ADV sequence (temporal
# adverb immediately followed by manner adverb) is rare; expect high surprisal
# at the first non-context token and elevated entropy in adverbial contexts.

message("--- Surprising: 'Hier soir, étrangement, il pleuvait des cordes.' ---")
dt_hard <- parse_one("Hier soir, étrangement, il pleuvait des cordes.")
res_hard <- pos_surprisal(dt_hard, pos_model, exclude_pos = EXCLUDE_POS,
                          use_sentence_boundaries = USE_BOUNDARIES)

print(res_hard$token_surprisal |>
  as_tibble() |>
  left_join(as_tibble(dt_hard)[, c("token_id", "token")], by = "token_id") |>
  select(token, upos, pos_surprisal, pos_entropy, pos_entropy_reduction) |>
  mutate(across(where(is.numeric), \(x) round(x, 2))))

# ── 3c. A nominalized / heavy-NP sentence ─────────────────────────────────────
#
# "La décision du conseil d'administration a été contestée."
# DET NOUN ADP NOUN ADP NOUN AUX AUX VERB — long nominal chain before the
# verb creates repeated ADP→DET→NOUN transitions (low surprisal there) but
# the double auxiliary sequence (AUX AUX VERB) is much rarer.

message("--- Nominalized: 'La décision du conseil d'administration a été contestée.' ---")
dt_nom <- parse_one("La décision du conseil d'administration a été contestée.")
res_nom <- pos_surprisal(dt_nom, pos_model, exclude_pos = EXCLUDE_POS,
                         use_sentence_boundaries = USE_BOUNDARIES)

print(res_nom$token_surprisal |>
  as_tibble() |>
  left_join(as_tibble(dt_nom)[, c("token_id", "token")], by = "token_id") |>
  select(token, upos, pos_surprisal, pos_entropy, pos_entropy_reduction) |>
  mutate(across(where(is.numeric), \(x) round(x, 2))))


# 4) Trigram vs. neural: full-context comparison ----
#
# The neural model conditions on the whole left context instead of just the
# two preceding tags, so it can disagree with the trigram model wherever
# longer-range structure matters — and it scores positions 1-2 that the
# trigram model leaves blank when use_sentence_boundaries = FALSE (not the
# case here, since USE_BOUNDARIES = TRUE for the toy sentences above).

compare_pos_backends <- function(dt) {
  trigram <- pos_surprisal(dt, pos_model, exclude_pos = EXCLUDE_POS,
                           use_sentence_boundaries = USE_BOUNDARIES)$token_surprisal
  neural  <- pos_surprisal_nn(dt, model_path = POS_LM_PATH,
                              exclude_pos = EXCLUDE_POS)$token_surprisal

  as_tibble(dt)[, c("token_id", "token")] |>
    left_join(as_tibble(trigram), by = "token_id") |>
    left_join(as_tibble(neural) |> select(token_id, pos_surprisal_nn, pos_entropy_nn),
              by = "token_id") |>
    select(token, upos, pos_surprisal, pos_surprisal_nn,
           pos_entropy, pos_entropy_nn) |>
    mutate(across(where(is.numeric), \(x) round(x, 2)))
}

message("--- Trigram vs. neural: 'Hier soir, étrangement, il pleuvait des cordes.' ---")
print(compare_pos_backends(dt_hard))


# 5) Full corpus scoring (on Viki-Wiki) ----

message("Loading parsed corpus...")
if (!file.exists("out/demo_parsed_tagged.Rds")) {
  stop("out/demo_parsed_tagged.Rds not found — run demos/demo_parse_tag.R first.",
       call. = FALSE)
}
dt_corpus <- readRDS("out/demo_parsed_tagged.Rds") |> as_tibble()

message("Scoring POS surprisal on full corpus (trigram)...")
corpus_res <- pos_surprisal(dt_corpus, pos_model, exclude_pos = EXCLUDE_POS,
                            use_sentence_boundaries = FALSE,
                            backoff_scale = NULL)

message("Scoring POS surprisal on full corpus (neural, full-context)...")
corpus_res_nn <- pos_surprisal_nn(dt_corpus, model_path = POS_LM_PATH,
                                  exclude_pos = EXCLUDE_POS)

dt_doc <- corpus_res$doc_surprisal |>
  as_tibble() |>
  left_join(as_tibble(corpus_res_nn$doc_surprisal), by = "doc_id") |>
  mutate(
    source  = if_else(str_detect(doc_id, "^viki_"), "Vikidia", "Wikipedia"),
    pair_id = str_extract(doc_id, "\\d+")
  )

message("Document-level means by source:")
dt_doc |>
  group_by(source) |>
  summarise(across(where(is.numeric) & !all_of("pair_id"), \(x) round(mean(x, na.rm = TRUE), 3))) |>
  print()


# 6) Visualisation ----
#
# Vikidia (simple French encyclopedia) uses more predictable syntactic patterns
# than Wikipedia. Documents are paired (same topic), so paired Cohen's d is used
# to remove between-topic variance — more sensitive than the unpaired estimate.
#
# Trigram and neural metrics are plotted side by side: if both backends agree
# on the Vikidia/Wikipedia direction, that is evidence the effect is not an
# artefact of either model's specific context window.

plot_faceted_boxplot(
  dt_doc, source,
  title    = "POS surprisal and entropy: Vikidia vs Wikipedia (trigram vs neural)"
)

