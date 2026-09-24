# alsi-ai / main_ai.R
#
# Python/LLM-backed demo pipeline, companion to ALSI's main.R. Parses the
# bundled Vikidia/Wikipedia demo corpus using ALSI (assumed cloned as a
# sibling directory, ../alsi) and then runs every AI feature family on it.
#
# Prerequisites:
#   - ../alsi cloned alongside this repo, with its demo corpus and models set up
#     (see ../alsi/main.R for the classic-only prerequisites)
#   - R packages: data.table, reticulate (installs Python deps automatically
#     via reticulate::py_require() on first use)
#
# Output:
#   out/demo_corpus_ai_features.Rds — AI feature families only
# ---------------------------------------------------------------------------

library(data.table)

ALSI_DIR <- "../alsi"

if (!dir.exists(ALSI_DIR)) {
  stop("main_ai.R | expected ALSI cloned at ", ALSI_DIR,
       ". Clone https://github.com/gloignon/alsi alongside this repo.")
}

# 1) Build the classic parsed corpus using ALSI ----
source(file.path(ALSI_DIR, "R/fnt_setup.R"),  encoding = "UTF-8")
source(file.path(ALSI_DIR, "R/fnt_corpus.R"), encoding = "UTF-8")
library(udpipe)
udmodel_french <- udpipe_load_model(file = file.path(ALSI_DIR, "models/french_gsd-remix_3.udpipe"))

# load_demo_corpus()/ensure_viki_wiki_demo_corpus() resolve paths relative to
# the working directory, so switch into ALSI_DIR just for this step.
old_wd <- setwd(ALSI_DIR)
dt_txt <- load_demo_corpus()
setwd(old_wd)

dt_parsed_raw <- parse_text(dt_txt, ud_model = file.path(ALSI_DIR, "models/french_gsd-remix_3.udpipe"))
parsed_corpus <- post_process_lexicon(dt_parsed_raw)

features <- list()

if (!reticulate::py_available(initialize = TRUE)) {
  stop("main_ai.R | Python not available. Install Python and rerun.")
}

library(reticulate)

# 2) Neural POS surprisal ----
source("R/fnt_pos_surprisal_nn.R", encoding = "UTF-8")
features$pos_surprisal_nn <- pos_surprisal_nn(
  parsed_corpus,
  model_path  = "models/pos_lm_fr_gsd_alsi.pt",
  exclude_pos = c("PUNCT", "SYM")
)

# 3) Embeddings and embedding coherence ----
source("R/fnt_embeddings.R", encoding = "UTF-8")
py_require(c("sentence-transformers", "torch", "numpy"))
features$embeddings <- corpus_embeddings(
  dt_corpus  = parsed_corpus,
  batch_size = 8
)

# 4) LLM surprisal and entropy ----
source("R/fnt_surprisal.R", encoding = "UTF-8")
py_require(c("transformers>=4.41,<5", "torch", "tokenizers", "numpy", "scipy"))

features$surprisal$mlm <- llm_surprisal_entropy(
  parsed_corpus,
  model_name = "almanach/moderncamembert-base",
  mode       = "mlm",
  batch_size = 8
)

features$surprisal$ar <- llm_surprisal_entropy(
  parsed_corpus,
  model_name       = "lightonai/pagnol-small",
  mode             = "ar",
  batch_size       = 8,
  add_prefix_space = TRUE
)

# 5) Save ----
dir.create("out", showWarnings = FALSE)
saveRDS(features, "out/demo_corpus_ai_features.Rds")
