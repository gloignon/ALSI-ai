# ALSI Demo: LLM Surprisal as a Writing Level Predictor (CEFLE corpus)
#
# "Surprisal" measures how unexpected each word is given its context.
# A language model predicts the probability of each word; surprisal is
# -log2(probability). A common word in a predictable context gets a low score;
# a rare or structurally odd word gets a high score.
#
# This demo uses the CEFLE transversal corpus: French essays written by
# Swedish learners of French at four proficiency levels (A–D, CEFR scale),
# plus a native-speaker baseline group (E). The question is: does mean
# surprisal decrease as writing level improves?
#
# Reference for the CEFLE corpus:
# Granfeldt, J., & Ågren, M. (2014). SLA developmental stages and teachers'
# assessment of written French: Exploring Direkt Profil as a diagnostic assessment 
# tool. Language Testing, 31(3), 285–305. https://doi.org/10.1177/0265532214526178
#
# In this demo you will:
#   1) load the CEFLE corpus from demo_corpora/cefle_corpus_texts.csv;
#   2) segment each document into sentences with split_sentences();
#   3) compute MLM surprisal with CamemBERT v2 (almanach/camembertv2-base),
#      letting the model's own tokenizer handle word boundaries;
#   4) aggregate to document level and compute descriptive statistics;
#   5) evaluate rank-ordering quality with Spearman rho and cross-validated QWK;
#   6) visualise the distributions across levels.
#
# Prerequisites:
#   - Python with transformers, torch, tokenizers, numpy (auto-installed)
#   - Best run in a fresh R session: reticulate locks parts of the Python
#     configuration once Python starts, so a session that already ran another
#     Python-based demo can fail to add packages.
#
# Runtime note: scoring 136 documents takes several minutes on CPU.
#   Results are cached in out/ so subsequent runs are instant.

library(tidyverse)
library(data.table)
library(reticulate)
library(nnet)  # multinom() is in base R's nnet package


# action = "add" lets this work even when Python was already initialised by
# another demo earlier in the same R session (the default action errors then).
py_require(c(
  "transformers>=4.41,<5",
  "torch",
  "tokenizers",
  "numpy"
), action = "add")

source("R/fnt_surprisal.R", encoding = "UTF-8")
source("../alsi/R/fnt_utility.R",   encoding = "UTF-8")

dir.create("out", showWarnings = FALSE)


# 1) Load CEFLE corpus ----
#
# The CSV has one row per document with columns: corpus, level, doc_id, text.
#
# Levels A–D are L2 (second-language learners at increasing CEFR stages);
# level E is the native-speaker (L1) baseline. The corpus was collected at
# Lund University as part of the CEFLE project.
#
# ALSI does not redistribute CEFLE (ELRA license prohibits redistribution).
# On first run the fetcher script downloads it directly from Lund University.

cefle_csv <- "demo_corpora/cefle_corpus_texts.csv"

if (!file.exists(cefle_csv)) {
  message(
    "\n--- CEFLE corpus not found — fetching from source ---\n",
    "\n",
    "The CEFLE transversal corpus is hosted by Lund University:\n",
    "  https://projekt.ht.lu.se/cefle/\n",
    "\n",
    "ALSI does not bundle this corpus. Each user must fetch it directly\n",
    "from Lund's servers. The corpus is for academic, non-commercial use\n",
    "only (ELRA standard license terms apply).\n",
    "\n",
    "By continuing you confirm that your use is non-commercial and academic.\n",
    "Fetching now..."
  )
  source("../alsi/R/artefact_builders/fetch_cefle_transversal.R", encoding = "UTF-8")
}

df_cefle <- read_csv(
  cefle_csv,
  col_types = cols(
    level  = col_character(),
    doc_id = col_character(),
    text   = col_character(),
    corpus = col_character()
  )
) |>
  mutate(level = factor(level, levels = c("A", "B", "C", "D", "E")))

message(
  "CEFLE loaded: ", nrow(df_cefle), " documents across levels ",
  paste(levels(df_cefle$level), collapse = ", ")
)

# Level breakdown: A is the lowest L2 stage, E is native French
print(count(df_cefle, level))


# 2) Segment into sentences ----
#
# llm_surprisal_entropy_raw() scores one sentence at a time, so we need to
# split each document into sentences first. split_sentences() from fnt_utility.R
# does this with a regex that splits on sentence-final punctuation followed by
# an uppercase letter — lightweight and fast, no external model required.
#
# We deliberately avoid pre-tokenizing words here: llm_surprisal_entropy_raw()
# passes each sentence as a raw string and lets the CamemBERT tokenizer decide
# word boundaries. This gives better scores than forcing an external tokenizer's
# segmentation onto the model.

dt_sentences <- df_cefle |>
  select(doc_id, text) |>
  purrr::pmap(function(doc_id, text) {
    sents <- split_sentences(text)
    tibble(
      doc_id      = doc_id,
      sentence_id = seq_along(sents),
      sentence    = sents
    )
  }) |>
  purrr::list_rbind() |>
  as.data.table()

message(
  "Segmented: ", nrow(dt_sentences), " sentences across ",
  n_distinct(dt_sentences$doc_id), " documents"
)


# 3) Compute MLM surprisal ----
#
# llm_surprisal_entropy_raw() passes each sentence as a raw string to the
# Python backend. The CamemBERT tokenizer splits it into subwords, the model
# scores each subword, and scores are aggregated back to word level using the
# tokenizer's own word_ids mapping.
#
# CamemBERT v2 is a masked language model (MLM): it sees both left and right
# context for each token simultaneously, which gives more reliable per-token
# probability estimates than a left-to-right model.

# The scoring backend selects the best available device automatically:
# MPS (Apple Silicon) > CUDA > CPU. To force CPU instead, set the
# environment variable before sourcing this script:
#   Sys.setenv(ALSIO_FORCE_CPU = "1")

cache_mlm <- "out/demo_cefle_mlm.Rds"

if (file.exists(cache_mlm)) {
  message("Loading cached surprisal scores...")
  dt_mlm <- readRDS(cache_mlm)
} else {
  message("Computing MLM surprisal (this may take several minutes)...")
  dt_mlm <- llm_surprisal_entropy_raw(
    dt_sentences, 
    model_name = "almanach/camembertv2-base",
    mode       = "mlm",
    batch_size = 8  # reduce to 1 if you get errors
  )
  saveRDS(dt_mlm, cache_mlm)
  message("Saved to ", cache_mlm)
}


# 4) Aggregate to document level ----
#
# We summarise each document to a single mean surprisal score by averaging
# over all its tokens. Tokens scored NA (sub-word fragments, punctuation
# handled differently across models) are excluded via na.rm = TRUE.
#
# We also retain n_tokens to give a sense of document length.

df_docs <- dt_mlm |>
  as_tibble() |>
  group_by(doc_id) |>
  summarise(
    mean_surprisal = mean(llm_surprisal, na.rm = TRUE),
    sd_surprisal   = sd(llm_surprisal,   na.rm = TRUE),
    n_tokens       = sum(!is.na(llm_surprisal)),
    .groups        = "drop"
  ) |>
  left_join(
    select(df_cefle, doc_id, level),
    by = "doc_id"
  ) |>
  mutate(
    level_num = as.integer(level),   # A=1 … E=5
    group     = if_else(level == "E", "L1 baseline (E)", "L2 (A–D)")
  )

message("\nDocument-level mean surprisal by level:")
df_docs |>
  group_by(level) |>
  summarise(
    n          = n(),
    mean       = round(mean(mean_surprisal), 2),
    sd         = round(sd(mean_surprisal),   2),
    median     = round(median(mean_surprisal), 2),
    .groups    = "drop"
  ) |>
  print()


# 5) Metrics: r², Spearman rho (full data), and cross-validated QWK ----
#
# Three complementary metrics, each measuring something slightly different:
#
#   Pearson r² — proportion of variance in the ordinal level explained by
#   mean surprisal. Computed on the full corpus: surprisal is a fixed score
#   with no fitted parameters, so CV adds nothing here.
#
#   Spearman rho (ρ) — same reasoning: rank correlation between the fixed
#   surprisal score and level number, full corpus.
#
#   QWK — requires converting the continuous surprisal score to discrete
#   level predictions via a trained classifier, which does fit parameters.
#   We use 5-fold CV so the classifier never sees held-out labels during
#   training. QWK is computed on the pooled out-of-fold predictions.


# 5-fold CV via cv_predict() from fnt_utility.R.
# cv_predict() handles stratified fold assignment and the train/test loop;
# we supply the fit and predict steps for our specific model.
# multinom() fits a softmax (multinomial logistic) classifier; trace = FALSE
# suppresses the iteration log that would otherwise flood the console.
df_oof <- cv_predict(
  df          = df_docs,
  strata_col  = "level",
  k           = 5L,
  fit_fn      = function(train) {
    nnet::multinom(
      level ~ mean_surprisal + sd_surprisal,
      data  = train,
      trace = FALSE
    )
  },
  predict_fn  = function(model, test) {
    as.integer(factor(
      predict(model, newdata = test, type = "class"),
      levels = c("A", "B", "C", "D", "E")
    ))
  }
)

# cv_predict() uses row position as .row_id, so add it to df_docs to join back.
df_docs  <- df_docs |> mutate(.row_id = row_number())
df_oof   <- df_oof |>
  left_join(select(df_docs, .row_id, level_num), by = ".row_id")

spearman_rho <- cor(df_docs$mean_surprisal, df_docs$level_num, method = "spearman")
pearson_r2   <- cor(df_docs$mean_surprisal, df_docs$level_num, method = "pearson")^2
qwk_cv       <- qwk(df_oof$level_num, df_oof$.pred)

# Kruskal-Wallis test: is the surprisal distribution the same across all
# five levels? A significant result (p < .05) means at least one level
# differs.
kw_test <- kruskal.test(mean_surprisal ~ level, data = df_docs)

# Cohen's d: L2 (A–D pooled) vs L1 (E).
x_l2 <- df_docs |> filter(level != "E") |> pull(mean_surprisal)
x_l1 <- df_docs |> filter(level == "E") |> pull(mean_surprisal)

cohens_d_l2_l1 <- (mean(x_l2) - mean(x_l1)) /
  sqrt((var(x_l2) * (length(x_l2) - 1) + var(x_l1) * (length(x_l1) - 1)) /
         (length(x_l2) + length(x_l1) - 2))

message("=== Surprisal as a level predictor (CamemBERT v2 MLM) ===\n")
message(sprintf("  Pearson r2   (surprisal ~ level):  %.3f", pearson_r2))
message(sprintf("  Spearman rho (surprisal ~ level):  %.3f", spearman_rho))
message(sprintf("  QWK (5-fold CV, multinomial):      %.3f", qwk_cv))
message(sprintf("  Kruskal-Wallis: H(%d) = %.2f, p = %s",
                kw_test$parameter,
                kw_test$statistic,
                if (kw_test$p.value < .001) "< .001" else sprintf("%.3f", kw_test$p.value)))
message(sprintf("  Cohen's d (L2 pooled vs L1):       %.3f", cohens_d_l2_l1))

# Confusion matrix: rows = true level, columns = predicted level.
# Good rank-ordering shows most counts near the diagonal.
message("Out-of-fold confusion matrix (rows = true, cols = predicted):")
print(table(
  true      = factor(df_oof$level_num, 1:5, labels = c("A","B","C","D","E")),
  predicted = factor(df_oof$.pred,     1:5, labels = c("A","B","C","D","E"))
))

# Pairwise: mean surprisal per level with Wilcoxon test vs E (native baseline).
message("Pairwise comparison vs L1 baseline (Wilcoxon rank-sum):")
df_docs |>
  filter(level != "E") |>
  group_by(level) |>
  summarise(
    mean    = round(mean(mean_surprisal), 2),
    p_vs_E  = wilcox.test(mean_surprisal, x_l1)$p.value,
    .groups = "drop"
  ) |>
  mutate(
    p_vs_E_fmt = if_else(p_vs_E < .001, "< .001", sprintf("%.3f", p_vs_E))
  ) |>
  select(level, mean, p_vs_E_fmt) |>
  print()


# 6) Visualisation: violin + boxplot by level ----
#
# Each panel shows the distribution of mean surprisal across documents at
# that level. The violin shape reveals the full distribution; the nested
# boxplot summarises the median and interquartile range; individual points
# show each document.
#
# Expected pattern (from left to right):
#   A (beginner) → highest surprisal (unusual words and structures)
#   B, C, D      → decreasing surprisal as proficiency grows
#   E (native)   → lowest surprisal (predictable, fluent French)
#
# The two fill colours distinguish L2 learner levels (blue) from the L1
# native-speaker baseline (salmon) so the contrast is immediately visible.

p_violin <- df_docs |>
  ggplot(aes(x = level, y = mean_surprisal, fill = group)) +
  geom_violin(alpha = 0.35, trim = TRUE, colour = NA) +
  geom_boxplot(width = 0.12, outlier.shape = NA, alpha = 0.85) +
  geom_jitter(
    aes(colour = group),
    width   = 0.06,
    size    = 1.0,
    alpha   = 0.5,
    show.legend = FALSE
  ) +
  scale_fill_manual(
    values = c("L2 (A–D)" = "#4472C4", "L1 baseline (E)" = "#C0504D")
  ) +
  scale_colour_manual(
    values = c("L2 (A–D)" = "#2a4a7f", "L1 baseline (E)" = "#8b2a2a")
  ) +
  labs(
    x     = "Writing level",
    y     = "Surprisal",
    fill  = NULL,
    title = "CEFLE: Mean surprisal by writing level",
    subtitle = sprintf(
      "R² = %.2f | Spearman ρ = %.2f | QWK (5-fold CV)= %.2f",
      pearson_r2, spearman_rho, qwk_cv
    )
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position   = "top",
    # legend.key.size   = unit(0.5, "lines"),
  )

print(p_violin)

# save a pdf of the plot
ggsave(
  "out/demo_cefle_surprisal_violin.pdf",
  plot = p_violin,
  width = 8,
  height = 6
)
