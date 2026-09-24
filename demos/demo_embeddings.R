# ALSI Demo: Embedding Coherence Features (Document-Level Comparison)
#
# Sentence embeddings represent each sentence as a point in a high-dimensional
# space where semantically similar sentences end up close together. ALSI uses
# a pre-trained French sentence encoder to embed each sentence, then derives
# structural features from how sentences relate to one another within a document.
#
# Features derived from embeddings:
#   thematic_dispersion    — mean cosine distance to the document centroid;
#                            high = topic wanders, low = stays on one theme
#   centroid_distance_sd   — variability of sentence-to-centroid distances
#   sequential_similarity  — mean cosine similarity between consecutive sentences;
#                            high = smooth topic flow
#   mean_semantic_gap      — mean cosine distance between consecutive sentences
#   max_semantic_gap       — largest single topic jump in the document
#   topic_drift            — mean cosine distance between consecutive 3-sentence blocks
#   mean_novelty           — mean cosine distance to the running centroid (so far)
#   n_topics               — estimated number of topic clusters (k = 1–5 by silhouette)
#   convexity              — Gärdenfors-style: pair midpoints stay near
#                            non-endpoint sentence embeddings; 1 = supported
#   blob_convexity         — proportion of pair midpoints inside the document's
#                            local support radius
#   segment_support        — local-scale support exp(-0.5 * normalised distance^2)
#   segment_occupancy      — proportion of midpoints within endpoint k-NN scale
#   local_convexity        — same midpoint support, only for consecutive pairs
#   local_blob_convexity   — local version of blob_convexity
#
# In this demo you will:
#   1) compute coherence features on the Viki-Wiki corpus;
#   2) compare Vikidia vs Wikipedia with Cohen's d effect sizes;
#   3) apply the same pipeline to ALECTOR (source vs simplified);
#   4) print a combined comparison table across both corpora.
#
# Prerequisites:
#   - alsi cloned alongside this repo (../alsi); run its demos/demo_parse_tag.R
#     first (creates ../alsi/out/demo_parsed_tagged.Rds)
#   - Python with sentence-transformers, transformers, torch, numpy, scipy
#     (auto-installed via py_require — you do not need to set up a venv)
#   - Internet access on first run to download the embedding model (~400 MB)
#   - Best run in a fresh R session: reticulate locks parts of the Python
#     configuration once Python starts, so a session that already ran another
#     Python-based demo can fail to add packages.


# 0) Setup ----

library(tidyverse)
library(data.table)
library(reticulate)  # R-Python bridge
library(effsize)     # for cohen.d()

# py_require() installs Python packages automatically if missing.
# action = "add" lets this work even when Python was already initialised by
# another demo earlier in the same R session (the default action errors then).
py_require(c(
  "transformers>=4.41,<5",
  "torch",
  "tokenizers",
  "sentence-transformers",
  "numpy",
  "scipy"
), action = "add")

ALSI_DIR <- "../alsi"
if (!dir.exists(ALSI_DIR)) {
  stop("demo_embeddings.R | expected ALSI cloned at ", ALSI_DIR,
       ". Clone https://github.com/gloignon/alsi alongside this repo.")
}

source(file.path(ALSI_DIR, "R/fnt_utility.R"), encoding = "UTF-8")
source("R/fnt_embeddings.R", encoding = "UTF-8")

dir.create("out", showWarnings = FALSE)

# The embedding model to use. This French document encoder is a good default;
# replace with any HuggingFace model compatible with sentence-transformers.
MODEL_NAME  <- "dangvantuan/french-document-embedding"
MODEL_MODE  <- "basic"
MODEL_LABEL <- "dangvantuan_french_document_embedding"

# Set to FALSE to recompute embeddings even if a cache file exists.
REUSE_EMBEDDINGS <- TRUE

# The coherence features we want to compare across groups.
feat_cols <- c(
  "emb_thematic_dispersion", "emb_centroid_distance_sd",
  "emb_sequential_similarity", "emb_mean_semantic_gap",
  "emb_max_semantic_gap", "emb_topic_drift",
  "emb_mean_novelty", "emb_n_topics",
  "emb_convexity", "emb_blob_convexity",
  "emb_segment_support", "emb_segment_occupancy",
  "emb_local_convexity", "emb_local_blob_convexity",
  "emb_local_segment_support", "emb_local_segment_occupancy"
)


# compare_groups() is defined in R/fnt_utility.R


# 1) Load Viki-Wiki corpus and compute embeddings ----
#
# corpus_embeddings() encodes each sentence and returns sentence-level
# and document-level embedding tables. The sentence-level table is what
# embedding_coherence() needs to compute structural features.

alsi_parsed_corpus <- file.path(ALSI_DIR, "out/demo_parsed_tagged.Rds")
if (!file.exists(alsi_parsed_corpus)) {
  stop(alsi_parsed_corpus, " not found — run demos/demo_parse_tag.R in ", ALSI_DIR, " first.",
       call. = FALSE)
}
dt_parsed_corpus <- readRDS(alsi_parsed_corpus)
message("Viki-Wiki corpus: ", n_distinct(dt_parsed_corpus$doc_id),
        " documents, ", nrow(dt_parsed_corpus), " tokens")

cache_sent <- sprintf("out/demo_emb_sent_%s.Rds", MODEL_LABEL)
cache_doc  <- sprintf("out/demo_emb_%s.Rds",      MODEL_LABEL)

if (REUSE_EMBEDDINGS && file.exists(cache_sent)) {
  message("Loading cached Viki-Wiki sentence embeddings")
  dt_sent <- readRDS(cache_sent)
} else {
  message("Computing Viki-Wiki embeddings (this may take a few minutes)...")
  emb    <- corpus_embeddings(dt_parsed_corpus, model_name = MODEL_NAME,
                              mode = MODEL_MODE, batch_size = 8)
  dt_sent <- emb$dt_sent_embeddings
  saveRDS(emb$dt_doc_embeddings, cache_doc)
  saveRDS(dt_sent, cache_sent)
  message("Saved to ", cache_sent)
}


# 2) Compute coherence features for Viki-Wiki ----
#
# embedding_coherence() takes sentence embeddings and returns one row per
# document with all structural features listed in the header above.

coherence_wv <- embedding_coherence(dt_sent) |>
  mutate(source = if_else(str_detect(doc_id, "^wiki_"), "wiki", "viki"))

message("Viki-Wiki coherence computed: ", nrow(coherence_wv), " documents")


# 3) Effect sizes for Viki-Wiki (unpaired) ----

effects_wv <- compare_groups(coherence_wv, "source", "wiki", "viki",
                             feat_cols, "wiki/viki", strip_prefix = "^emb_")


# 4) Download ALECTOR corpus ----
#
# 79 pairs of French texts for primary-school children: each pair has an
# original source text and a manually simplified target version.

alector_dir  <- "out/alector_corpus"
alector_base <- "https://raw.githubusercontent.com/gloignon/alector_corpus/master"
n_alector    <- 79

dir.create(alector_dir, recursive = TRUE, showWarnings = FALSE)

# crossing() produces all combinations of pair index × variant.
alector_files <- crossing(i = 0:(n_alector - 1), suffix = c("source", "target")) |>
  mutate(
    fname = sprintf("%03d_%s.txt", i, suffix),
    dest  = file.path(alector_dir, fname),
    url   = sprintf("%s/corpus/%s", alector_base, fname)
  )

# Download any files not already on disk.
alector_files |>
  filter(!file.exists(dest)) |>
  pwalk(function(url, dest, fname, ...) {
    tryCatch(download.file(url, dest, quiet = TRUE),
             error = function(e) warning(sprintf("Download failed: %s", fname)))
  })


# 5) Read ALECTOR texts ----
#
# read_sentences() reads a single .txt file and returns a tibble of sentences.
# purrr::map() applies it to every file; compact() drops NULLs (missing files).

dt_alector <- alector_files |>
  filter(file.exists(dest)) |>
  mutate(doc_id = sprintf("alector_%s_%02d", suffix, i)) |>
  purrr::pmap(function(dest, doc_id, ...) read_sentences(dest, doc_id)) |>
  purrr::compact() |>
  bind_rows()

message("ALECTOR: ", n_distinct(dt_alector$doc_id), " documents, ",
        nrow(dt_alector), " sentences")


# 6) Embed ALECTOR and compute coherence ----

cache_alector_sent <- sprintf("out/demo_emb_alector_sent_%s.Rds", MODEL_LABEL)

if (REUSE_EMBEDDINGS && file.exists(cache_alector_sent)) {
  message("Loading cached ALECTOR sentence embeddings")
  dt_alector_sent <- readRDS(cache_alector_sent)
} else {
  message("Computing ALECTOR embeddings...")
  emb_alector     <- corpus_embeddings(dt_alector, model_name = MODEL_NAME,
                                       mode = MODEL_MODE, batch_size = 8)
  dt_alector_sent <- emb_alector$dt_sent_embeddings
  saveRDS(emb_alector$dt_doc_embeddings,
          sprintf("out/demo_emb_alector_%s.Rds", MODEL_LABEL))
  saveRDS(dt_alector_sent, cache_alector_sent)
  message("Saved to ", cache_alector_sent)
}

# pair_id links each source text to its corresponding target text.
coherence_al <- embedding_coherence(dt_alector_sent) |>
  mutate(
    source  = if_else(str_detect(doc_id, "_source_"), "source", "target"),
    pair_id = str_remove(doc_id, "alector_(source|target)_")
  )

message("ALECTOR coherence computed: ", nrow(coherence_al), " documents")


# 7) Effect sizes for ALECTOR (paired) ----
#
# Because texts are paired by topic, using paired = TRUE removes between-topic
# variance and gives a more sensitive (and more honest) effect size estimate.

effects_al <- compare_groups(
  coherence_al, "source", "source", "target",
  feat_cols, "alector",
  paired = TRUE, pair_col = "pair_id", strip_prefix = "^emb_"
)


# 8) Combined comparison table ----
#
# Positive d: first group (wiki / source) has higher values.
# Negative d: second group (viki / target) has higher values.

effects_both <- bind_rows(effects_wv, effects_al) |>
  mutate(
    d   = round(d, 3),
    p   = round(p, 4),
    sig = case_when(
      p < .001 ~ "***",
      p < .01  ~ "**",
      p < .05  ~ "*",
      TRUE     ~ ""
    )
  ) |>
  arrange(corpus, desc(abs(d)))

message("\nCoherence feature comparison (Cohen's d + Wilcoxon p):\n")
print(effects_both, n = Inf)


# 9) Boxplots ----

# Viki-Wiki: unpaired comparison (each document is independent)
plot_faceted_boxplot(
  coherence_wv, source, all_of(feat_cols),
  title   = "Embedding Coherence: Vikidia vs Wikipedia",
  y_lab   = NULL
)

# ALECTOR: paired by topic — pair_col removes between-topic variance from the
# Cohen's d labels, giving a more honest estimate of the simplification effect.
plot_faceted_boxplot(
  coherence_al, source, all_of(feat_cols),
  title    = "Embedding Coherence: ALECTOR source vs target",
  y_lab    = NULL,
  paired   = TRUE,
  pair_col = "pair_id"
)
