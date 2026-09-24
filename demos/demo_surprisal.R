# ALSI Demo: LLM Surprisal and Entropy (Masked Language Model)
#
# "Surprisal" measures how unexpected each word is given its context.
# A masked language model (MLM, e.g. CamemBERT) predicts the probability
# of each word given the words around it. Surprisal = -log2(probability):
# a common word in a typical context gets low surprisal; a rare or
# unexpected word gets high surprisal.
#
# "Entropy" is a complementary measure: it captures how uncertain the model
# is about what word *could* come next, regardless of which word actually did.
# High entropy = many plausible continuations; low entropy = constrained context.
#
# Simpler texts (e.g. children's texts) tend to use more predictable words →
# lower mean surprisal. This demo uses the ALECTOR corpus: 79 pairs of French
# texts for children, each consisting of an original source text and a
# simplified target version.
#
# In this demo you will:
#   1) download the ALECTOR corpus (if not already cached);
#   2) parse it with UDPipe;
#   3) compute MLM surprisal and entropy with CamemBERT;
#   4) aggregate to document level and compare source vs target;
#   5) visualise with boxplots and print a summary statistics table.
#
# Prerequisites:
#   - Python with transformers, torch, tokenizers, numpy (auto-installed via py_require)
#   - Internet access for first download of corpus and model
#   - models/french_gsd-remix_3.udpipe
#   - Best run in a fresh R session: reticulate locks parts of the Python
#     configuration once Python starts, so a session that already ran another
#     Python-based demo can fail to add packages.
#
# Note: computing surprisal on 158 documents takes several minutes on CPU;
#   much faster with a GPU. Results are cached in out/ to avoid recomputing.

library(tidyverse)
library(data.table)
library(reticulate)  # bridges R and Python
library(udpipe)

# py_require() automatically installs the listed Python packages if they are
# not already present. You do not need to manage a virtual environment manually.
# action = "add" is required (rather than the default "set") because Python
# may already be initialised by a prior demo in the same R session — the
# default action is only allowed before initialisation.
py_require(c(
  "transformers>=4.41,<5",
  "torch",
  "tokenizers",
  "numpy"
), action = "add")

source("R/fnt_surprisal.R",       encoding = "UTF-8")
source("R/fnt_top_predictions.R", encoding = "UTF-8")
source("../alsi/R/fnt_corpus.R",          encoding = "UTF-8")
source("../alsi/R/fnt_utility.R",         encoding = "UTF-8")



# 0) Basic functionality on a single demo sentence ----
#
# Before running the full corpus, we illustrate the three core measures on one
# hand-picked sentence so you can see exactly what each function returns: one
# row per word, with a surprisal and an entropy value.
#
# We use `llm_surprisal_entropy_raw()`, which takes a data.table of *raw*
# sentences (columns doc_id, sentence_id, sentence) and lets the model's own
# tokenizer decide word boundaries. This is the recommended entry point for
# scoring natural text (see the discussion in section 3).

demo_sentence <- data.table::data.table(
  doc_id      = "demo",
  sentence_id = 1L,
  sentence    = "La petite chatte grise dort dans le panier d'osier."
)

# --- (a) MLM surprisal + entropy -------------------------------------------
#
# A masked language model (CamemBERT) uses BOTH left and right context to
# predict each word. Surprisal is high for words the model finds unexpected;
# entropy is high where many continuations were plausible.

message("\n0a) MLM surprisal + entropy (CamemBERT, both-side context)")
demo_mlm <- llm_surprisal_entropy_raw(
  demo_sentence,
  model_name = "almanach/moderncamembert-base",
  mode       = "mlm"
)
print(as_tibble(demo_mlm) |>
        select(token, llm_surprisal, llm_entropy, llm_subword_n))

# --- (b) AR surprisal + entropy --------------------------------------------
#
# An autoregressive model (GPT-style) predicts each word from LEFT context
# only — left to right, as in reading. This is the classic psycholinguistic
# notion of surprisal. Note: with no `context`, the first word has no left
# context and its surprisal is reported as NA (see the roxygen note).
#
# We use PAGnol-small, a small French GPT-style model, so the demo stays
# lightweight. This model's tokenizer needs `add_prefix_space = TRUE` to align
# word boundaries correctly.

message("\n0b) AR surprisal + entropy (PAGnol-small, left-context only)")
demo_ar <- llm_surprisal_entropy_raw(
  demo_sentence,
  model_name       = "lightonai/pagnol-small",
  mode             = "ar",
  add_prefix_space = TRUE
)
print(as_tibble(demo_ar) |>
        select(token, llm_surprisal, llm_entropy, llm_subword_n))

# --- (c) The entropy modes -------------------------------------------------
#
# When a word is split into several sub-word tokens, its single `llm_entropy`
# value has to be derived from the per-subword entropies. The `entropy`
# argument controls which per-word entropy is reported:
#   "onset"     — entropy at the word's FIRST subtoken (DEFAULT; uncertainty
#                 *before* committing to the word — the approximation used by
#                 several recent papers)
#   "mean"      — average over the word's subwords
#   "sum"       — total over the word's subwords
#   "offset"    — entropy at the word's LAST subtoken (uncertainty about what
#                 comes *after* the word — used in the corpus analysis below)
#   "none"      — skip entropy entirely; `llm_entropy` is NA and the entropy
#                 computation is not run (use when you only need surprisal)
#
# Surprisal is unaffected by `entropy`; only the entropy column changes.
# We reprocess the same sentence under each mode and join the entropy columns
# side by side so the effect is easy to see. ("none" is left out here since it
# would just yield an all-NA column.)

message("\n0c) Same sentence, four entropy modes (MLM)")
entropy_modes <- c("onset", "mean", "sum", "offset")

demo_entropy <- entropy_modes |>
  purrr::map(function(mode_i) {
    llm_surprisal_entropy_raw(
      demo_sentence,
      model_name = "almanach/moderncamembert-base",
      mode       = "mlm",
      entropy    = mode_i
    ) |>
      as_tibble() |>
      transmute(token_id, token, "entropy_{mode_i}" := llm_entropy, llm_subword_n)
  }) |>
  purrr::reduce(dplyr::left_join,
                by = c("token_id", "token", "llm_subword_n"))

# llm_subword_n is identical across modes (it depends on tokenization, not on
# the entropy mode), so we keep a single copy and show it as the last column.
print(demo_entropy |> select(-token_id, -llm_subword_n, llm_subword_n))

# --- (d) Top-k predictions: what did the model expect? ---------------------
#
# Surprisal tells you *how* surprised the model was at a word, but not *what*
# it expected instead. `llm_word_predictions()` answers that: it masks each
# word's whole span (turning the sentence into a fill-in-the-blank) and reports
# the whole words the model judged most probable in the slot — the same idea as
# a "cloze" test, where a reader guesses a missing word from context.
#
# Concretely, for "Les randonneurs fatigués _____ ... la montagne enneigée.",
# the model sees every position blanked in turn and offers its top-k fill-ins.
# We show k = 5 candidates per word, plus:
#   surprisal      — bits of surprise for the word actually written
#   word_prob      — the model's probability for the written word (shown as "-"
#                    when the word is split into several sub-word pieces, which
#                    a single mask cannot spell; see (e) for how to recover it)
#   actual_in_topk — TRUE (shown as "*") when the written word was itself among
#                    the model's top 5 guesses
#
# `print_word_predictions()` lays this out as a fixed-width table.

message("\n0d) Top-5 whole-word predictions per position (cloze-style, MLM)")
demo_topk <- llm_word_predictions(
  sentence   = "Les randonneurs fatigués grimpaient lentement la montagne enneigée.",
  k          = 5L,
  model_name = "almanach/moderncamembert-base"
)
print_word_predictions(demo_topk)

# --- (e) Recovering multi-piece candidates by beam search ------------------
#
# The single-mask view in (d) can only propose words that are a SINGLE
# vocabulary token, so longer or rarer words (which the tokenizer splits into
# pieces) never appear and their `word_prob` is "-". `llm_beam_word_predictions()`
# lifts that limit: it refills each blank with one to `max_pieces` masks and
# assembles multi-piece words left-to-right by beam search. It is slower, but
# every word now gets a probability. Compare the rows for "randonneurs",
# "fatigués" and "grimpaient" against (d): those split words now carry a
# probability and appear among their own top-5 candidates.

message("\n0e) Top-5 predictions incl. multi-piece words (beam search, MLM)")
demo_beam <- llm_beam_word_predictions(
  sentence   = "Les randonneurs fatigués grimpaient lentement la montagne enneigée.",
  k          = 5L,
  max_pieces = 3L,
  model_name = "almanach/moderncamembert-base"
)
print_word_predictions(demo_beam)

# DEMO ON THE ALECTOR CORPUS -----

# 1) Download ALECTOR (if not already present) ----
#
# ALECTOR contains 79 paired texts: each pair has an original (source)
# and a simplified version (target) written for primary-school children.
# Files are named 000_source.txt, 000_target.txt, 001_source.txt, etc.

alector_dir  <- "out/alector_corpus"
alector_base <- "https://raw.githubusercontent.com/gloignon/alector_corpus/master"
n_alector    <- 79

dir.create(alector_dir, recursive = TRUE, showWarnings = FALSE)

# Build a table listing every file: 79 pairs × 2 variants = 158 rows.
alector_files <- tibble(
  i      = rep(0:(n_alector - 1), each = 2),
  suffix = rep(c("source", "target"), times = n_alector)
) |>
  mutate(
    fname  = sprintf("%03d_%s.txt", i, suffix),
    dest   = file.path(alector_dir, fname),
    url    = sprintf("%s/corpus/%s", alector_base, fname),
    doc_id = sprintf("alector_%s_%02d", suffix, i)
  )

# Only download files that are not already on disk.
alector_files |>
  filter(!file.exists(dest)) |>
  pwalk(function(url, dest, fname, ...) {
    tryCatch(
      download.file(url, dest, quiet = TRUE),
      error = function(e) warning(sprintf("Download failed: %s", fname))
    )
  })


# 2) Parse ALECTOR with UDPipe ----
#
# Parsing is slow (several minutes for 158 documents), so we cache the
# result. On subsequent runs this block is skipped and the cache is loaded.

cache_alector_parsed <- "out/alector_parsed.Rds"

if (file.exists(cache_alector_parsed)) {
  message("Loading cached ALECTOR parsed corpus")
  dt_alector <- readRDS(cache_alector_parsed)
} else {
  message("Parsing ALECTOR corpus with UDPipe (this may take a few minutes)...")

  # Read every file into a data.table with one row per document.
  # rowwise() + mutate() allows reading each file's text inline.
  dt_alector_txt <- alector_files |>
    filter(file.exists(dest)) |>
    rowwise() |>
    mutate(text = paste(readLines(dest, encoding = "UTF-8", warn = FALSE), collapse = " ")) |>
    ungroup() |>
    select(doc_id, text) |>
    as_tibble()

  message(nrow(dt_alector_txt), " documents loaded")

  n_cores        <- max(1, parallel::detectCores() - 1)
  dt_alector_raw <- parse_text(dt_alector_txt, n_cores = n_cores)
  dt_alector     <- post_process_lexicon(dt_alector_raw)
  saveRDS(dt_alector, cache_alector_parsed)
  message("Saved to ", cache_alector_parsed)
}

message("ALECTOR parsed: ", nrow(dt_alector), " tokens, ",
        n_distinct(dt_alector$doc_id), " documents")


# 3) Compute MLM surprisal on ALECTOR ----

dt_alector_sentences <- dt_alector |>
  as_tibble() |>
  distinct(doc_id, sentence_id, sentence) |>
  filter(!is.na(sentence), nzchar(sentence)) |>
  as.data.table()

cache_al_mlm <- "out/demo_surprisal_al_mlm_onset.Rds"

if (file.exists(cache_al_mlm)) {
  message("Loading cached ALECTOR MLM surprisal scores")
  dt_alector_mlm <- readRDS(cache_al_mlm)
} else {
  message("Computing MLM surprisal on raw sentences (may take several minutes)...")
  dt_alector_mlm <- llm_surprisal_entropy_raw(
    dt_alector_sentences,
    model_name = "almanach/moderncamembert-base",
    mode       = "mlm",
    entropy    = "onset",  # onset entropy: uncertainty before the word (the
                           # default; approximation used by several recent papers)
    batch_size = 10  # reduce to 1 if you get memory errors
  )
  saveRDS(dt_alector_mlm, cache_al_mlm)
  message("Saved to ", cache_al_mlm)
}


# 4) Aggregate to document level ----
#
# We average surprisal and entropy over all tokens in each document.
# pair_id lets us link each source text to its corresponding target text
# for paired analyses.

df_docs <- dt_alector_mlm |>
  as_tibble() |>
  mutate(
    source  = if_else(str_detect(doc_id, "_source_"), "source", "target"),
    pair_id = as.integer(str_remove(doc_id, "alector_(source|target)_"))
  ) |>
  group_by(doc_id, source, pair_id) |>
  summarise(
    llm_surprisal = mean(llm_surprisal, na.rm = TRUE),
    llm_entropy   = mean(llm_entropy,   na.rm = TRUE),
    .groups = "drop"
  )

print(df_docs)


# 5) Faceted boxplots ----
#
# Source texts should have higher surprisal and entropy than target texts
# if simplification makes language more predictable.
# The plot_faceted_boxplot() utility handles paired labeling automatically.

source("../alsi/R/fnt_utility.R", encoding = "UTF-8")

df_docs |>
  rename(Surprisal = llm_surprisal, Entropy = llm_entropy) |>
  plot_faceted_boxplot(
    source, c(Surprisal, Entropy),
    title = "ALECTOR: Original vs Simplified (CamemBERT MLM)",
    y_lab = "Mean per document",
    notch = TRUE
  )


# 6) Summary statistics table ----
#
# Paired Cohen's d (pooled SD) and a paired Wilcoxon p-value come straight
# from the compare_groups() utility, so the demo does not re-implement effect
# sizes. NOTE: compare_groups() uses effsize::cohen.d(paired = TRUE), i.e. the
# pooled-SD Cohen's d — NOT mean(diff)/sd(diff), which is d_z and inflates the
# estimate for positively correlated pairs.
#
# Spearman r (how consistently one group outranks the other across pairs) is
# demo-specific and computed here.

# Effect size (paired Cohen's d) + paired Wilcoxon p, one row per metric.
df_effects <- compare_groups(
  df           = df_docs,
  grp_col      = "source",
  grp_a        = "source",
  grp_b        = "target",
  feat_cols    = c("llm_surprisal", "llm_entropy"),
  corpus_label = "ALECTOR",
  paired       = TRUE,
  pair_col     = "pair_id"
) |>
  transmute(
    metric   = recode(feature, llm_surprisal = "Surprisal", llm_entropy = "Entropy"),
    cohens_d = round(d, 2),
    p        = if_else(p < 0.001, "< .001", sprintf("%.3f", p))
  )

# Wide-by-pair for the descriptive means and Spearman r.
df_paired <- df_docs |>
  pivot_longer(
    cols = c(llm_surprisal, llm_entropy),
    names_to  = "metric",
    values_to = "value"
  ) |>
  mutate(metric = recode(metric, llm_surprisal = "Surprisal", llm_entropy = "Entropy")) |>
  pivot_wider(id_cols = c(pair_id, metric), names_from = source, values_from = value)

df_summary <- df_paired |>
  group_by(metric) |>
  summarise(
    mean_source = round(mean(source, na.rm = TRUE), 2),
    mean_target = round(mean(target, na.rm = TRUE), 2),
    spearman_r  = round(cor(source, target, method = "spearman"), 2),
    n_pairs     = n(),
    .groups = "drop"
  ) |>
  left_join(df_effects, by = "metric")

message("\n=== ALECTOR: Source vs Target (paired, CamemBERT MLM) ===\n")
print(df_summary |> select(metric, mean_source, mean_target, cohens_d, spearman_r, p, n_pairs))
