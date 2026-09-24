# fnt_corpus_backends.R
#
# spaCy and Trankit dependency parsing backends, as alternatives to alsi's
# udpipe-based parse_text(). Both run through Python via reticulate and are
# kept out of the alsi (classic) package for that reason.
#
# Usage: source alsi's R/ files first (for data.table/normalize helpers and
# post_process_lexicon()/reparse_copular_etre()), then source this file.

# ── Shared Python setup ───────────────────────────────────────────────────────
#
# reticulate::py_require() can only set `python_version` (and the package
# list as a whole) BEFORE Python initializes; once a backend has triggered
# initialization (e.g. via source_python()), later calls may only `action =
# "add"` new packages — passing `python_version` again throws, even with an
# identical value. Since spaCy and Trankit can both run in the same session
# and Trankit needs Python 3.10, we declare the full combined requirement
# exactly once, on whichever backend initializes Python first.
.alsi_py_state <- new.env(parent = emptyenv())
.alsi_py_state$declared <- FALSE

.alsi_py_require <- function() {
  if (!isTRUE(.alsi_py_state$declared)) {
    reticulate::py_require(
      packages = c("spacy>=3.8.14,<3.9.0", "click",
                   "trankit==1.1.2", "adapters==1.0.0",
                   "transformers>=4.40,<4.44", "huggingface_hub<0.26",
                   "numpy<2", "syntok"),
      python_version = "3.10"
    )
    .alsi_py_state$declared <- TRUE
  }
  return(invisible(NULL))
}

# ── spaCy backend ─────────────────────────────────────────────────────────────

#' Parse Text with spaCy
#'
#' Python-backed alternative to \code{alsi::parse_text()}. Requires
#' \pkg{reticulate} and a spaCy model directory.
#'
#' @param txt Either a data.frame with columns \code{doc_id} and \code{text},
#'   or a character vector of texts.
#' @param spacy_model Path to the spaCy model directory.
#' @param show_progress Logical; if TRUE, shows a progress bar or messages.
#' @param chunk_size Number of documents per chunk.
#' @returns A \code{data.table} in CoNLL-U format (one row per token).
parse_text_spacy <- function(txt, spacy_model = "models/spacy_fr_gsd_alsi_v1",
                             show_progress = TRUE, chunk_size = 50L) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")
  }

  .alsi_py_require()

  reticulate::source_python(
    system.file("py/spacy_backend.py", package = "alsiai", mustWork = FALSE) |>
      (\(p) if (nzchar(p)) p else here::here("py/spacy_backend.py"))()
  )

  normalize_input <- function(x) {
    if (is.data.frame(x)) {
      if (!all(c("doc_id", "text") %in% names(x)))
        stop("parse_text_spacy | data.frame must contain columns: doc_id, text")
      dt <- as.data.table(x[, c("doc_id", "text")])
      dups <- dt$doc_id[duplicated(dt$doc_id)]
      if (length(dups) > 0)
        stop("parse_text_spacy | duplicate doc_id values: ",
             paste(unique(dups), collapse = ", "),
             "\nEach document must have a unique doc_id.")
      return(dt)
    }
    if (is.character(x)) {
      ids <- if (!is.null(names(x))) names(x) else paste0("doc_", seq_along(x))
      dt  <- data.table(doc_id = ids, text = unname(x))
      dups <- dt$doc_id[duplicated(dt$doc_id)]
      if (length(dups) > 0)
        stop("parse_text_spacy | duplicate doc_id values: ",
             paste(unique(dups), collapse = ", "),
             "\nEach document must have a unique doc_id.")
      return(dt)
    }
    stop("parse_text_spacy | txt must be a data.frame with doc_id/text or a character vector")
  }

  txt_dt <- normalize_input(txt)

  if (!dir.exists(spacy_model)) {
    stop("parse_text_spacy | spaCy model not found at: ", spacy_model,
         "\nPlace the model folder under models/ or pass spacy_model = <path>.")
  }

  n <- nrow(txt_dt)
  if (show_progress)
    message("parse_text_spacy | ", n, " text(s) to process")

  chunks <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  pb <- if (show_progress && n > 1L)
    utils::txtProgressBar(min = 0, max = n, style = 3)
  else
    NULL
  on.exit(if (!is.null(pb)) close(pb), add = TRUE)

  rows <- vector("list", length(chunks))
  for (i in seq_along(chunks)) {
    idx <- chunks[[i]]
    rows[[i]] <- reticulate::py$spacy_parse_texts(
      texts      = as.list(txt_dt$text[idx]),
      doc_ids    = as.list(txt_dt$doc_id[idx]),
      model_path = spacy_model
    )
    if (!is.null(pb)) utils::setTxtProgressBar(pb, idx[length(idx)])
  }
  rows <- unlist(rows, recursive = FALSE)

  if (length(rows) == 0L) return(data.table())

  dt <- data.table::rbindlist(lapply(rows, as.list), use.names = TRUE, fill = TRUE)

  dt[, token_id      := as.integer(token_id)]
  dt[, head_token_id := as.integer(head_token_id)]
  dt[, feats         := ifelse(feats == "NA", NA_character_, feats)]
  dt[, paragraph_id  := 1L]
  dt[, term_id       := .I]

  data.table::setcolorder(dt, intersect(
    c("doc_id", "paragraph_id", "sentence_id", "term_id",
      "token_id", "token", "lemma", "upos", "xpos", "feats",
      "head_token_id", "dep_rel"),
    names(dt)
  ))

  return(dt)
}

# ── Trankit backend ────────────────────────────────────────────────────────────

.trankit_ensure_model <- function(model_dir) {
  default <- "models/trankit_fr_v1"
  path <- if (!is.null(model_dir)) model_dir else default
  if (!dir.exists(path)) {
    stop(
      "parse_text_trankit | Trankit model not found at: ", path, "\n",
      "Place the trankit_fr_v1 folder under models/ or pass trankit_model = <path>."
    )
  }
  path
}

#' Parse Text with Trankit
#'
#' Python-backed alternative to \code{alsi::parse_text()}. Requires
#' \pkg{reticulate} and will download XLM-RoBERTa base (~1.1 GB) on first use.
#'
#' @param txt Either a data.frame with columns \code{doc_id} and \code{text},
#'   or a character vector of texts.
#' @param trankit_model Path to the Trankit model directory (the folder
#'   containing \code{xlm-roberta-base/}).
#' @param trankit_category Category name used when training the Trankit model.
#'   Default \code{"customized-mwt"}.
#' @param trankit_gpu Logical; use GPU for Trankit inference. Default FALSE.
#' @param show_progress Logical; if TRUE, shows a progress bar or messages.
#' @param chunk_size Number of documents per chunk.
#' @returns A \code{data.table} in CoNLL-U format (one row per token).
parse_text_trankit <- function(txt, trankit_model = NULL,
                               trankit_category = "customized-mwt",
                               trankit_gpu = FALSE, show_progress = TRUE,
                               chunk_size = 50L) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")
  }

  .alsi_py_require()

  reticulate::source_python(
    system.file("py/trankit_backend.py", package = "alsiai", mustWork = FALSE) |>
      (\(p) if (nzchar(p)) p else here::here("py/trankit_backend.py"))()
  )

  normalize_input <- function(x) {
    if (is.data.frame(x)) {
      if (!all(c("doc_id", "text") %in% names(x)))
        stop("parse_text_trankit | data.frame must contain columns: doc_id, text")
      dt <- as.data.table(x[, c("doc_id", "text")])
      dups <- dt$doc_id[duplicated(dt$doc_id)]
      if (length(dups) > 0)
        stop("parse_text_trankit | duplicate doc_id values: ",
             paste(unique(dups), collapse = ", "),
             "\nEach document must have a unique doc_id.")
      return(dt)
    }
    if (is.character(x)) {
      ids <- if (!is.null(names(x))) names(x) else paste0("doc_", seq_along(x))
      dt  <- data.table(doc_id = ids, text = unname(x))
      dups <- dt$doc_id[duplicated(dt$doc_id)]
      if (length(dups) > 0)
        stop("parse_text_trankit | duplicate doc_id values: ",
             paste(unique(dups), collapse = ", "),
             "\nEach document must have a unique doc_id.")
      return(dt)
    }
    stop("parse_text_trankit | txt must be a data.frame with doc_id/text or a character vector")
  }

  txt_dt    <- normalize_input(txt)
  model_dir <- .trankit_ensure_model(trankit_model)
  n         <- nrow(txt_dt)

  # Warm up the model before the progress bar so loading messages appear first.
  reticulate::py$trankit_load(model_dir, trankit_category, trankit_gpu)

  if (show_progress)
    message("parse_text_trankit | ", n, " text(s) to process")

  chunks <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  pb <- if (show_progress && n > 1L)
    utils::txtProgressBar(min = 0, max = n, style = 3)
  else
    NULL
  on.exit(if (!is.null(pb)) close(pb), add = TRUE)

  rows <- vector("list", length(chunks))
  for (i in seq_along(chunks)) {
    idx <- chunks[[i]]
    rows[[i]] <- reticulate::py$trankit_parse_texts(
      texts     = as.list(txt_dt$text[idx]),
      doc_ids   = as.list(txt_dt$doc_id[idx]),
      model_dir = model_dir,
      category  = trankit_category,
      gpu       = trankit_gpu
    )
    if (!is.null(pb)) utils::setTxtProgressBar(pb, idx[length(idx)])
  }
  rows <- unlist(rows, recursive = FALSE)

  if (length(rows) == 0L) return(data.table())

  dt <- data.table::rbindlist(lapply(rows, as.list), use.names = TRUE, fill = TRUE)

  dt[, token_id      := as.integer(token_id)]
  dt[, head_token_id := as.integer(head_token_id)]
  dt[, feats         := ifelse(feats == "NA", NA_character_, feats)]
  dt[, paragraph_id  := 1L]
  dt[, term_id       := .I]

  data.table::setcolorder(dt, intersect(
    c("doc_id", "paragraph_id", "sentence_id", "term_id",
      "token_id", "token", "lemma", "upos", "xpos", "feats",
      "head_token_id", "dep_rel"),
    names(dt)
  ))

  return(dt)
}

# ── syntok sentence segmenter ─────────────────────────────────────────────────

#' Segment texts into sentences with syntok (Python)
#'
#' A lightweight alternative to \code{alsi::parse_text()} when only sentence
#' boundaries are needed and installing a UDPipe model is undesirable.
#' Calls \code{py/segment_sentences.py} via \pkg{reticulate}.
#'
#' @param txt Either a \code{data.frame}/\code{data.table} with columns
#'   \code{doc_id} and \code{text}, or a character vector of texts.
#' @param min_sent_len Integer. Trailing sentences shorter than this many
#'   tokens are merged into the preceding sentence. Set to 0 to disable.
#'   Default 4.
#' @param merge_colon_semicolon Logical; if TRUE, sentences ending with
#'   \code{:} or \code{;} are merged with the following sentence when safe.
#'   Default FALSE.
#' @param max_prev_len,max_next_len,max_merged_len Token-length guards for
#'   the colon/semicolon merge heuristic. See \code{segment_sentences.py}.
#' @returns A \code{data.table} with columns \code{doc_id},
#'   \code{sentence_id} (restarts at 1 per document), and \code{sentence}.
segment_sentences_syntok <- function(txt,
                                     min_sent_len = 4L,
                                     merge_colon_semicolon = FALSE,
                                     max_prev_len = 35L,
                                     max_next_len = 35L,
                                     max_merged_len = 60L) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")
  }
  .alsi_py_require()
  reticulate::source_python(
    system.file("py/segment_sentences.py", package = "alsiai",
                mustWork = FALSE) |>
      (\(p) if (nzchar(p)) p else "py/segment_sentences.py")()
  )

  normalize_input <- function(x) {
    if (is.data.frame(x)) {
      if (!all(c("doc_id", "text") %in% names(x))) {
        stop("segment_sentences_syntok | data.frame must have columns: doc_id, text")
      }
      dt <- as.data.table(x[, c("doc_id", "text")])
      dups <- dt$doc_id[duplicated(dt$doc_id)]
      if (length(dups) > 0)
        stop("segment_sentences_syntok | duplicate doc_id values: ",
             paste(unique(dups), collapse = ", "),
             "\nEach document must have a unique doc_id.")
      return(dt)
    }
    if (is.character(x)) {
      ids <- if (!is.null(names(x))) names(x) else paste0("doc_", seq_along(x))
      dt  <- data.table(doc_id = ids, text = unname(x))
      dups <- dt$doc_id[duplicated(dt$doc_id)]
      if (length(dups) > 0)
        stop("segment_sentences_syntok | duplicate doc_id values: ",
             paste(unique(dups), collapse = ", "),
             "\nEach document must have a unique doc_id.")
      return(dt)
    }
    stop("segment_sentences_syntok | txt must be a data.frame with doc_id/text or a character vector")
  }

  txt_dt <- normalize_input(txt)

  rows <- vector("list", nrow(txt_dt))
  for (i in seq_len(nrow(txt_dt))) {
    did  <- txt_dt$doc_id[i]
    text <- txt_dt$text[i]
    sents <- reticulate::py$segment_text_syntok(
      text,
      min_sent_len         = as.integer(min_sent_len),
      merge_colon_semicolon = merge_colon_semicolon,
      max_prev_len         = as.integer(max_prev_len),
      max_next_len         = as.integer(max_next_len),
      max_merged_len       = as.integer(max_merged_len)
    )
    if (length(sents) == 0L) next
    rows[[i]] <- data.table(
      doc_id      = did,
      sentence_id = seq_along(sents),
      sentence    = as.character(sents)
    )
  }

  return(data.table::rbindlist(rows, use.names = TRUE))
}
