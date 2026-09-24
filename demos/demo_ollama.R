# ALSI Demo: Ollama API — Generate, Summarize, and Verify Statements
#
# This demo shows how to use a local LLM (via Ollama) for structured text
# processing tasks: extracting facts, summarizing, and checking consistency.
# All queries are stateless — the model does not remember previous turns.
#
# Workflow (3 passes over the same Wikipedia texts):
#   Pass 1: Ask the LLM to extract 4 true statements from each full text.
#   Pass 2: Ask the LLM to write a short summary (2–3 sentences) of each text.
#   Pass 3: Ask the LLM to verify each statement against the *summary* (not the
#           full text). This tests how much factual content survives compression.
#
# Results are written incrementally to CSV files in out/. If you stop and
# restart, the script resumes where it left off (no duplicate queries).
#
# Why Ollama?
#   Ollama runs models locally on your machine — no API key, no cost, no data
#   leaving your computer. This is important for corpora with confidentiality
#   requirements (student texts, clinical data, etc.).
#
# Prerequisites:
#   - Ollama installed and running (https://ollama.com)
#   - Model pulled before running this script:
#       ollama pull gemma3:4b
#     (run this once in a terminal — downloads ~3 GB)
#   - demo_corpora/viki_wiki.zip — demo .txt corpus; wiki_ prefix docs are used
#
#
# Using a non-Ollama backend (openai protocol)
#   set OLLAMA_ENDPOINT to your server's address and api to "openai",
#   and model to the name of the model as defined in your config.
#   e.g. if your LM Studio is at http://localhost:8080, set:
#       OLLAMA_ENDPOINT <- "http://localhost:8080"
#       api = "openai"
#       model = "gemma3-4b"


# 0) Setup ----

library(data.table)

source("../alsi/R/fnt_corpus.R",  encoding = "UTF-8")
source("R/fnt_ollama.R",  encoding = "UTF-8")
source("../alsi/R/fnt_setup.R", encoding = "UTF-8")

# Address (IP/hostname + port) of the Ollama server. Change this if Ollama
# is running on a nonstandard port or on a different machine than the one
# running this script.
OLLAMA_ENDPOINT <- "http://localhost:11434"


# 1) Smoke test — verify Ollama is responding ----
#
# Before running the full pipeline, send one simple question.
# If this fails, check that Ollama is running and the model is pulled.

dt_test <- data.table(question = "Quelle est la capitale de la France ?")
dt_test  <- ollama_generate(
  data                 = dt_test,
  user_prompt_template = "{question}",
  system_prompt        = "Réponds en une phrase.",
  model                = "gemma3:4b",
  temperature          = 0,
  num_ctx              = 256,
  force_restart        = TRUE,
  endpoint             = OLLAMA_ENDPOINT
)

message("Smoke test:")
message("  Q: ", dt_test$question[1])
message("  A: ", dt_test$ollama_response[1])

if (is.na(dt_test$ollama_response[1]) || !nzchar(dt_test$ollama_response[1])) {
  stop("Smoke test failed: no response from Ollama. ",
       "Make sure Ollama is running and 'gemma3:4b' is pulled.")
}


# 2) Load a few Wikipedia texts ----
#
# We use only N_DOCS documents to keep the demo fast.
# Adjust upward to run on the full corpus.

N_DOCS <- 5

dt_corpus <- load_demo_corpus()
dt_wiki   <- dt_corpus[grepl("^wiki_", doc_id)][1:N_DOCS]

message("Demo corpus: ", nrow(dt_wiki), " documents")
print(dt_wiki[, .(doc_id, nchar = nchar(text))])


# 3) Pass 1: Extract true statements from full texts ----
#
# We ask the model to produce exactly 4 numbered statements per text.
# The strict system prompt ("Ne produis rien d'autre") is important — without
# it, chat models often add preambles and formatting that break parsing.

system_prompt_statements <- paste(
  "Voici un texte encyclopédique en français.",
  "Tu dois produire exactement 4 affirmations vraies à propos du texte.",
  "Numérote-les de 1 à 4. Ne produis rien d'autre."
)

dt_statements <- ollama_generate(
  data                 = dt_wiki,
  user_prompt_template = "Voici le texte :\n\n{text}",
  system_prompt        = system_prompt_statements,
  model                = "gemma3:4b",
  temperature          = 0.3,    # a little randomness so statements aren't identical
  num_ctx              = 4096,
  output_file          = "out/demo_ollama_statements.csv",
  force_restart        = TRUE,
  endpoint             = OLLAMA_ENDPOINT
)

# parse_numbered_list() is defined in R/fnt_ollama.R.
# It extracts items from a numbered list response into a character vector.
# do.call(rbind, ...) stacks one vector per document into an n_docs × 4 matrix.
parsed <- do.call(rbind, purrr::map(dt_statements$ollama_response, parse_numbered_list))
for (i in 1:4) dt_statements[[paste0("statement_", i)]] <- parsed[, i]

message("Parsed statements:")
print(dt_statements[, .(doc_id, statement_1, statement_2, statement_3, statement_4)])


# 4) Pass 2: Summarize each text ----
#
# The summaries will be used in Pass 3 instead of the full texts.
# This simulates a "lossy compression" scenario: how much factual content
# survives a 2–3 sentence summary?

dt_summ <- ollama_generate(
  data                 = dt_wiki,
  user_prompt_template = "Voici le texte :\n\n{text}",
  system_prompt        = paste(
    "Tu es un assistant spécialisé en résumé de texte.",
    "On te donne un texte encyclopédique en français.",
    "Produis un résumé en 2 à 3 phrases. Ne produis rien d'autre."
  ),
  model                = "gemma3:4b",
  temperature          = 0.3,
  num_ctx              = 4096,
  output_file          = "out/demo_ollama_summaries.csv",
  force_restart        = TRUE,
  endpoint             = OLLAMA_ENDPOINT
)

message("Summaries:")
for (i in seq_len(nrow(dt_summ))) {
  resp <- dt_summ$ollama_response[i]
  if (is.na(resp) || !nzchar(resp)) resp <- "[no response]"
  message("=== ", dt_summ$doc_id[i], " ===\n", resp, "\n")
}


# 5) Pass 3: Verify statements against summaries ----
#
# Each statement (from Pass 1) is evaluated against the corresponding summary
# (from Pass 2). The model has NOT seen the full text here — only the summary.
#
# A verdict of VRAI = the statement is recoverable from the summary.
# A verdict of FAUX = the statement was lost in compression.
# UNCLEAR = the model gave an ambiguous response (counted separately).

# Reshape from wide (4 statement columns) to long (one row per statement).
dt_eval <- melt(
  dt_statements[, .(doc_id, statement_1, statement_2, statement_3, statement_4)],
  id.vars      = "doc_id",
  measure.vars = paste0("statement_", 1:4),
  variable.name = "statement_num",
  value.name    = "statement"
)[!is.na(statement)]

dt_eval <- dt_eval |> mutate(statement_num = as.integer(gsub("statement_", "", statement_num)))

# Join summaries onto the evaluation table.
dt_eval <- merge(dt_eval, dt_summ[, .(doc_id, summary = ollama_response)], by = "doc_id")

dt_eval <- ollama_generate(
  data                 = dt_eval,
  user_prompt_template = paste0(
    "Résumé du texte source :\n{summary}\n\n",
    "Affirmation :\n{statement}"
  ),
  system_prompt = paste(
    "Tu es un vérificateur d'affirmations.",
    "On te donne le résumé d'un texte source et une affirmation.",
    "Tu dois répondre UNIQUEMENT par VRAI ou FAUX.",
    "VRAI si l'affirmation est cohérente avec le résumé, FAUX sinon.",
    "Ne produis rien d'autre qu'un seul mot : VRAI ou FAUX."
  ),
  model       = "gemma3:4b",
  temperature = 0.0,   # deterministic for verification tasks
  num_ctx     = 4096,
  output_file = "out/demo_ollama_eval.csv",
  force_restart = TRUE,
  endpoint    = OLLAMA_ENDPOINT
)


# 6) Report results ----

# Parse VRAI / FAUX from the model response.
# fcase() is data.table's multi-condition if/else (like dplyr's case_when).
dt_eval <- dt_eval |> mutate(
  verdict = case_when(
    grepl("VRAI", ollama_response, ignore.case = TRUE) ~ "VRAI",
    grepl("FAUX", ollama_response, ignore.case = TRUE) ~ "FAUX",
    TRUE ~ "UNCLEAR"
  )
)

message("Evaluation results (statements vs summaries):")
print(dt_eval[, .(doc_id, statement_num, statement, verdict)])

n_expected  <- nrow(dt_statements) * 4
n_parsed    <- sum(!is.na(unlist(dt_statements[, paste0("statement_", 1:4), with = FALSE])))
n_vrai      <- sum(dt_eval$verdict == "VRAI")
n_faux      <- sum(dt_eval$verdict == "FAUX")
n_unclear   <- sum(dt_eval$verdict == "UNCLEAR")
n_evaluated <- nrow(dt_eval)

message("\nSummary:")
message(sprintf("  Statements parsed:              %d/%d (%.0f%%)",
                n_parsed, n_expected, 100 * n_parsed / n_expected))
message(sprintf("  Recovered from summary (VRAI):  %d/%d (%.0f%%)",
                n_vrai, n_evaluated, 100 * n_vrai / n_evaluated))
message(sprintf("  Lost in summary (FAUX):         %d/%d (%.0f%%)",
                n_faux, n_evaluated, 100 * n_faux / n_evaluated))
message(sprintf("  Unclear:                        %d/%d",
                n_unclear, n_evaluated))
