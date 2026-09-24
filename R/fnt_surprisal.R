
# Valid per-word entropy modes; kept in sync with `_ENTROPY_MODES` in
# py/llm_scoring.py. "none" skips entropy entirely.
.ENTROPY_MODES <- c("onset", "mean", "sum", "offset", "none")

# Normalise the `entropy` argument: NULL means "unspecified" and falls back to
# the default ("onset"); any other value must be one of .ENTROPY_MODES, else we
# stop rather than silently substituting a mode the caller did not ask for.
.normalize_entropy <- function(entropy) {
  if (is.null(entropy)) {
    return("onset")
  }
  if (length(entropy) != 1L || !is.character(entropy) ||
      !(entropy %in% .ENTROPY_MODES)) {
    stop("`entropy` must be NULL or one of: ",
         paste(.ENTROPY_MODES, collapse = ", "), call. = FALSE)
  }
  return(entropy)
}

#' Load a language model for token-level scoring
#'
#' Initialises a masked or autoregressive language model via the Python
#' \code{llm_scoring.py} backend. The model is cached in the Python session.
#'
#' @param model_name HuggingFace model identifier.
#' @param mode Either \code{"mlm"} (masked LM) or \code{"ar"} (autoregressive).
#' @param use_fast Logical; use the fast tokenizer if available.
#' @param trust_remote_code Logical; allow remote code execution for the model.
#' @param add_prefix_space Logical or NULL; add a leading space to tokens.
#' @param force_fast Logical; bypass AutoTokenizer and load
#'   \code{PreTrainedTokenizerFast} directly from \code{tokenizer.json}. Use
#'   this when a model repo declares a slow \code{tokenizer_class} but ships
#'   only \code{tokenizer.json} (e.g., \code{almanach/camembertv2-base}),
#'   which would otherwise cause AutoTokenizer to fall through to
#'   character-level tokenization.
#' @returns Invisible TRUE on success.
load_llm_scorer <- function(model_name = "almanach/moderncamembert-base",
                            mode = "mlm",
                            use_fast = TRUE,
                            trust_remote_code = TRUE,
                            add_prefix_space = NULL,
                            force_fast = FALSE) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")
  }
  message(sprintf("Loading %s LLM model '%s' (this may take a while on first run)...", mode, model_name))
  reticulate::py_require(c("torch", "transformers"), action = "add")
  reticulate::source_python("py/llm_scoring.py")
  load_llm_model(
    model_name,
    mode = mode,
    use_fast = use_fast,
    trust_remote_code = trust_remote_code,
    add_prefix_space = add_prefix_space,
    force_fast = isTRUE(force_fast)
  )
  invisible(TRUE)
}

#' Compute token-level LLM surprisal and entropy
#'
#' Scores each sentence with a masked or autoregressive language model and
#' returns per-token surprisal, entropy, and sub-word token count.
#'
#' @param dt_corpus A \code{data.table} with \code{doc_id},
#'   \code{sentence_id}, and \code{token}, plus one of \code{term_id},
#'   \code{vrai_token_id}, or \code{token_id} (preferred in that order) to
#'   establish token order within each sentence.
#' @param model_name HuggingFace model identifier.
#' @param mode Either \code{"mlm"} or \code{"ar"}.
#' @param context Optional context string prepended to each sentence.
#' @param batch_size Batch size for MLM scoring (0 = auto).
#' @param temperature Softmax temperature (default 1.0).
#' @param use_fast Logical; use the fast tokenizer.
#' @param trust_remote_code Logical; allow remote code for the model.
#' @param add_prefix_space Logical or NULL; add a leading space to tokens.
#' @param pll_mode Character; PLL scoring mode for MLM models. Either
#'   \code{"original"} (Salazar et al. 2020) or \code{"within_word_l2r"}
#'   (Kauf & Ivanova 2023). Ignored for AR models.
#' @param entropy Character; which per-word entropy to report, i.e. how a
#'   multi-subword word's \code{llm_entropy} is derived from its subword
#'   entropies. One of \code{"onset"} (default, entropy of the word's first
#'   subtoken, the approximation used by several recent papers), \code{"mean"}
#'   (mean across subwords), \code{"sum"} (sum across subwords),
#'   \code{"offset"} (entropy of the word's last subtoken), or \code{"none"}
#'   (skip entropy entirely; \code{llm_entropy} is \code{NA} and the entropy
#'   computation is not run). Passing \code{NULL} uses the default.
#' @param drop_mwt_parts Logical (default \code{TRUE}); drop the expanded
#'   parts of multi-word tokens (MWTs) before scoring. udpipe /
#'   \code{post_process_lexicon} emit both the surface contraction ("des",
#'   \code{token_id} "5-6") and its syntactic parts ("de", "le",
#'   \code{token_id} "5", "6"). The LLM reads surface text, so keeping all
#'   three would score every contraction three times. When \code{TRUE} the
#'   surface range row is kept and its parts removed; the parts never appear
#'   in surface text. Set \code{FALSE} only if \code{dt_corpus} contains no
#'   MWT range rows (already flattened upstream).
#' @returns The input \code{data.table} augmented with \code{llm_surprisal},
#'   \code{llm_entropy}, and \code{llm_subword_n} columns.
llm_surprisal_entropy <- function(dt_corpus,
                                  model_name = "almanach/moderncamembert-base",
                                  mode = "mlm",
                                  context = NULL,
                                  batch_size = 0,
                                  temperature = 1.0,
                                  use_fast = TRUE,
                                  trust_remote_code = TRUE,
                                  add_prefix_space = NULL,
                                  pll_mode = "original",
                                  entropy = NULL,
                                  drop_mwt_parts = TRUE) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")
  }
  if (!("doc_id" %in% names(dt_corpus)) ||
      !("sentence_id" %in% names(dt_corpus)) ||
      !("token" %in% names(dt_corpus))) {
    stop("dt_corpus must contain columns: doc_id, sentence_id, token")
  }
  entropy <- .normalize_entropy(entropy)

  dt <- data.table::as.data.table(data.table::copy(dt_corpus))

  # Choose a stable ordering within sentence. Prefer the integer row
  # counters: token_id is character in udpipe output, and sorting it
  # lexicographically scrambles sentences with 10+ tokens ("10" < "2").
  if ("term_id" %in% names(dt)) {
    order_col <- "term_id"
  } else if ("vrai_token_id" %in% names(dt)) {
    order_col <- "vrai_token_id"
  } else if ("token_id" %in% names(dt)) {
    order_col <- "token_id"
    if (is.character(dt$token_id)) {
      # Numeric prefix ("30-31" -> 30); the sort is stable, so MWT range
      # rows stay ahead of their parts.
      dt[, .token_order := suppressWarnings(
        as.integer(sub("-.*$", "", token_id)))]
      order_col <- ".token_order"
    }
  } else {
    stop("No token order column found. Expected term_id, vrai_token_id, or token_id.")
  }

  data.table::setorderv(dt, c("doc_id", "sentence_id", order_col))
  if (order_col == ".token_order") {
    dt[, .token_order := NULL]
  }

  # Multi-word-token (MWT) de-duplication. udpipe/post_process_lexicon keeps
  # BOTH the surface range row ("des", token_id "5-6") AND its expanded parts
  # ("de" token_id "5", "le" token_id "6"). The LLM sees surface text, so
  # scoring all three triples-counts every contraction. Keep the range row
  # (the surface form the model reads) and drop its parts. Parts are the rows
  # whose integer token_id falls inside a hyphenated range in the same
  # sentence.
  if (isTRUE(drop_mwt_parts) && "token_id" %in% names(dt) &&
      any(grepl("-", as.character(dt$token_id)))) {
    dt[, .tid_chr := as.character(token_id)]
    dt[, .is_range := grepl("-", .tid_chr)]
    dt[, .tid_int := suppressWarnings(as.integer(.tid_chr))]
    dt[, .rng_lo := suppressWarnings(as.integer(sub("-.*$", "", .tid_chr)))]
    dt[, .rng_hi := suppressWarnings(as.integer(sub("^.*-", "", .tid_chr)))]
    dt[, .drop_part := FALSE]
    # For each sentence, flag non-range rows whose token_id is covered by a
    # range row.
    dt[, .drop_part := {
      lo <- .rng_lo[.is_range]
      hi <- .rng_hi[.is_range]
      covered <- rep(FALSE, .N)
      if (length(lo)) {
        for (r in seq_along(lo)) {
          covered <- covered | (!.is_range & !is.na(.tid_int) &
                                  .tid_int >= lo[r] & .tid_int <= hi[r])
        }
      }
      covered
    }, by = .(doc_id, sentence_id)]
    dt <- dt[.drop_part == FALSE]
    dt[, c(".tid_chr", ".is_range", ".tid_int",
           ".rng_lo", ".rng_hi", ".drop_part") := NULL]
  }

  dt[, token_index := seq_len(.N), by = .(doc_id, sentence_id)]

  # In AR mode without context, the first word of each sentence is scored
  # from a BOS-only fallback (no left context) and is not comparable to the
  # rest; report it as NA.
  no_context <- mode == "ar" && (is.null(context) || !nzchar(context))

  # Load Python scoring backend (only if not already loaded)
  reticulate::source_python("py/llm_scoring.py")
  py_state <- tryCatch(get_llm_state(), error = function(e) NULL)
  py_model <- if (!is.null(py_state)) py_state$model_name else NULL
  py_mode <- if (!is.null(py_state)) py_state$mode else NULL
  py_prefix_space <- if (!is.null(py_state)) py_state$add_prefix_space else NULL

  if (is.null(py_model) ||
      is.null(py_mode) ||
      py_model != model_name ||
      py_mode != mode ||
      is.null(py_prefix_space) ||
      (!is.null(add_prefix_space) && isTRUE(add_prefix_space) != isTRUE(py_prefix_space))) {
    load_llm_model(
      model_name,
      mode = mode,
      use_fast = use_fast,
      trust_remote_code = trust_remote_code,
      add_prefix_space = add_prefix_space
    )
  }

  dt_sentences <- dt[, .(tokens = list(token)), by = .(doc_id, sentence_id)]
  total_sentences <- nrow(dt_sentences)

  results <- vector("list", total_sentences)
  start_time <- Sys.time()

  print_status <- function(processed, total, start_time) {
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    speed <- processed / elapsed * 60  # sentences per minute
    remaining <- (total - processed) / speed * 60  # seconds
    cat(sprintf("\rProcessed %d/%d sentences (%.1f sent/min) — ~%.1f min left     ",
                processed, total, speed, remaining / 60), sep = "")
    flush.console()
  }

  for (i in seq_len(total_sentences)) {
    sent_tokens <- dt_sentences$tokens[[i]]
    sent_tokens <- as.character(sent_tokens)
    if (anyNA(sent_tokens)) {
      sent_tokens[is.na(sent_tokens)] <- ""
    }
    doc_id <- dt_sentences$doc_id[[i]]
    sentence_id <- dt_sentences$sentence_id[[i]]

    score <- tryCatch({
      if (mode == "mlm") {
        score_masked_lm_tokens(
          tokens = sent_tokens,
          temperature = temperature,
          batch_size = batch_size,
          context_text = context,
          pll_mode = pll_mode,
          entropy = entropy
        )
      } else if (mode == "ar") {
        score_autoregressive_tokens(
          tokens = sent_tokens,
          temperature = temperature,
          context_text = context,
          entropy = entropy
        )
      } else {
        stop("mode must be one of: 'mlm', 'ar'")
      }
    }, error = function(e) {
      warning(sprintf("LLM scoring failed for doc_id = %s, sentence_id = %s: %s",
                      doc_id, sentence_id, e$message))
      NULL
    })

    if (!is.null(score)) {
      n_tokens <- length(sent_tokens)
      llm_surprisal <- unlist(score$word_surprisals)
      llm_entropy <- unlist(score$word_entropies)
      if (no_context && n_tokens > 0) {
        llm_surprisal[1] <- NA_real_
        llm_entropy[1] <- NA_real_
      }
      results[[i]] <- data.table::data.table(
        doc_id = doc_id,
        sentence_id = sentence_id,
        token_index = seq_len(n_tokens),
        llm_surprisal = llm_surprisal,
        llm_entropy = llm_entropy,
        llm_subword_n = unlist(score$word_token_counts)
      )
    }

    if (i %% 3 == 0 || i == total_sentences) {
      print_status(i, total_sentences, start_time)
    }
  }

  cat("\nDone!\n")

  results <- Filter(Negate(is.null), results)
  if (length(results) == 0) {
    dt[, `:=`(
      llm_surprisal = NA_real_,
      llm_entropy = NA_real_,
      llm_subword_n = NA_integer_
    )]
    dt_out <- dt
  } else {
    dt_scores <- data.table::rbindlist(results, use.names = TRUE, fill = TRUE)
    dt_out <- merge(
      dt,
      dt_scores,
      by = c("doc_id", "sentence_id", "token_index"),
      all.x = TRUE
    )
  }

  dt_out[, token_index := NULL]

  return(dt_out)
}


#' Compute sentence-level LLM surprisal and entropy
#'
#' Takes a data.table of sentences (not parsed tokens), tokenises each sentence
#' on whitespace, scores with the LLM backend, and returns one row per sentence
#' with mean surprisal and entropy.
#'
#' @param dt_sentences A \code{data.table} with \code{doc_id},
#'   \code{sentence_id}, and \code{sentence}.
#' @param model_name HuggingFace model identifier.
#' @param mode Either \code{"mlm"} or \code{"ar"}.
#' @param context Optional context string prepended to each sentence.
#' @param batch_size Batch size for MLM scoring (0 = auto).
#' @param temperature Softmax temperature (default 1.0).
#' @param use_fast Logical; use the fast tokenizer.
#' @param trust_remote_code Logical; allow remote code for the model.
#' @param add_prefix_space Logical or NULL; add a leading space to tokens.
#' @param pll_mode Character; PLL scoring mode. See \code{llm_surprisal_entropy}.
#' @param entropy Character; which per-word entropy to report (or
#'   \code{"none"} to skip it). See \code{llm_surprisal_entropy}.
#' @returns A \code{data.table} with \code{doc_id}, \code{sentence_id},
#'   \code{sentence}, \code{llm_surprisal}, and \code{llm_entropy}
#'   (sentence-level means).
llm_surprisal_entropy_sentences <- function(dt_sentences,
                                            model_name = "almanach/moderncamembert-base",
                                            mode = "mlm",
                                            context = NULL,
                                            batch_size = 0,
                                            temperature = 1.0,
                                            use_fast = TRUE,
                                            trust_remote_code = TRUE,
                                            add_prefix_space = NULL,
                                            pll_mode = "original",
                                            entropy = NULL) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")
  }
  required <- c("doc_id", "sentence_id", "sentence")
  missing <- setdiff(required, names(dt_sentences))
  if (length(missing)) {
    stop(sprintf("dt_sentences must contain columns: %s", paste(required, collapse = ", ")))
  }
  entropy <- .normalize_entropy(entropy)

  dt <- data.table::as.data.table(data.table::copy(dt_sentences))
  dt_tok <- dt[, .(token = unlist(strsplit(sentence, "\\s+")),
                    token_id = seq_along(unlist(strsplit(sentence, "\\s+")))),
               by = .(doc_id, sentence_id)]

  dt_scored <- llm_surprisal_entropy(
    dt_tok,
    model_name = model_name,
    mode = mode,
    context = context,
    batch_size = batch_size,
    temperature = temperature,
    use_fast = use_fast,
    trust_remote_code = trust_remote_code,
    add_prefix_space = add_prefix_space,
    pll_mode = pll_mode,
    entropy = entropy
  )

  dt_result <- dt_scored[, .(
    llm_surprisal = mean(llm_surprisal, na.rm = TRUE),
    llm_entropy   = mean(llm_entropy,   na.rm = TRUE)
  ), by = .(doc_id, sentence_id)]

  dt_result <- merge(
    dt[, .(doc_id, sentence_id, sentence)],
    dt_result,
    by = c("doc_id", "sentence_id"),
    all.x = TRUE
  )

  data.table::setorderv(dt_result, c("doc_id", "sentence_id"))
  dt_result
}


#' Compute word-level LLM surprisal and entropy on raw sentences
#'
#' Passes each sentence to the Python backend as a raw string (no
#' pre-tokenization on the R side). The LLM tokenizer's own pre-tokenizer
#' decides word boundaries, the model scores every subword, and the
#' backend aggregates subwords back to those words via
#' \code{encoding.word_ids()}. Returns one row per LLM-tokenizer word.
#'
#' @param dt_sentences A \code{data.table} with \code{doc_id},
#'   \code{sentence_id}, and \code{sentence}.
#' @param model_name HuggingFace model identifier.
#' @param mode Either \code{"mlm"} or \code{"ar"}.
#' @param context Optional context string prepended to each sentence.
#' @param batch_size Batch size for MLM scoring (0 = auto).
#' @param temperature Softmax temperature.
#' @param use_fast Logical; use the fast tokenizer.
#' @param trust_remote_code Logical; allow remote code for the model.
#' @param add_prefix_space Logical or NULL; add a leading space to tokens.
#' @param pll_mode Character; PLL scoring mode. See \code{llm_surprisal_entropy}.
#' @param entropy Character; which per-word entropy to report (or
#'   \code{"none"} to skip it). See \code{llm_surprisal_entropy}.
#' @returns A \code{data.table} with \code{doc_id}, \code{sentence_id},
#'   \code{token_id} (word index within the sentence), \code{token} (the
#'   word as the tokenizer sees it), \code{llm_surprisal},
#'   \code{llm_entropy}, and \code{llm_subword_n} (one row per word).
llm_surprisal_entropy_raw <- function(dt_sentences,
                                      model_name = "almanach/moderncamembert-base",
                                      mode = "mlm",
                                      context = NULL,
                                      batch_size = 0,
                                      temperature = 1.0,
                                      use_fast = TRUE,
                                      trust_remote_code = TRUE,
                                      add_prefix_space = NULL,
                                      pll_mode = "original",
                                      entropy = NULL) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")
  }
  required <- c("doc_id", "sentence_id", "sentence")
  missing <- setdiff(required, names(dt_sentences))
  if (length(missing)) {
    stop(sprintf("dt_sentences must contain columns: %s", paste(required, collapse = ", ")))
  }
  entropy <- .normalize_entropy(entropy)

  dt <- data.table::as.data.table(data.table::copy(dt_sentences))
  data.table::setorderv(dt, c("doc_id", "sentence_id"))

  # In AR mode without context, the first word of each sentence is scored
  # from a BOS-only fallback (no left context) and is not comparable to the
  # rest; report it as NA.
  no_context <- mode == "ar" && (is.null(context) || !nzchar(context))

  reticulate::py_require(c("torch", "transformers"), action = "add")
  reticulate::source_python("py/llm_scoring.py")
  py_state <- tryCatch(get_llm_state(), error = function(e) NULL)
  py_model <- if (!is.null(py_state)) py_state$model_name else NULL
  py_mode  <- if (!is.null(py_state)) py_state$mode       else NULL
  py_prefix_space <- if (!is.null(py_state)) py_state$add_prefix_space else NULL

  if (is.null(py_model) || is.null(py_mode) ||
      py_model != model_name || py_mode != mode ||
      is.null(py_prefix_space) ||
      (!is.null(add_prefix_space) && isTRUE(add_prefix_space) != isTRUE(py_prefix_space))) {
    load_llm_model(
      model_name,
      mode = mode,
      use_fast = use_fast,
      trust_remote_code = trust_remote_code,
      add_prefix_space = add_prefix_space
    )
  }

  total_sentences <- nrow(dt)
  results <- vector("list", total_sentences)
  start_time <- Sys.time()

  for (i in seq_len(total_sentences)) {
    sentence <- dt$sentence[i]
    doc_id <- dt$doc_id[i]
    sentence_id <- dt$sentence_id[i]

    score <- tryCatch({
      if (mode == "mlm") {
        score_masked_lm_tokens(
          tokens              = sentence,
          temperature         = temperature,
          batch_size          = batch_size,
          context_text        = context,
          is_split_into_words = FALSE,
          pll_mode            = pll_mode,
          entropy         = entropy
        )
      } else if (mode == "ar") {
        score_autoregressive_tokens(
          tokens              = sentence,
          temperature         = temperature,
          context_text        = context,
          is_split_into_words = FALSE,
          entropy         = entropy
        )
      } else {
        stop("mode must be one of: 'mlm', 'ar'")
      }
    }, error = function(e) {
      warning(sprintf("LLM scoring failed for doc_id = %s, sentence_id = %s: %s",
                      doc_id, sentence_id, e$message))
      NULL
    })

    if (!is.null(score)) {
      n_words <- length(score$word_surprisals)
      if (n_words > 0) {
        llm_surprisal <- unlist(score$word_surprisals)
        llm_entropy <- unlist(score$word_entropies)
        if (no_context) {
          llm_surprisal[1] <- NA_real_
          llm_entropy[1] <- NA_real_
        }
        results[[i]] <- data.table::data.table(
          doc_id        = doc_id,
          sentence_id   = sentence_id,
          token_id      = seq_len(n_words),
          token         = unlist(score$word_tokens),
          llm_surprisal = llm_surprisal,
          llm_entropy   = llm_entropy,
          llm_subword_n = unlist(score$word_token_counts),
          tokenizer_oov = unlist(score$word_oov)
        )
      }
    }

    if (i %% 10L == 0L || i == total_sentences) {
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      if (elapsed > 0) {
        speed <- i / elapsed * 60
        cat(sprintf("\rProcessed %d/%d sentences (%.1f sent/min)     ",
                    i, total_sentences, speed))
        flush.console()
      }
    }
  }
  cat("\nDone!\n")

  results <- Filter(Negate(is.null), results)
  if (length(results) == 0) {
    return(data.table::data.table(
      doc_id = character(), sentence_id = integer(), token_id = integer(),
      token = character(), llm_surprisal = numeric(),
      llm_entropy = numeric(), llm_subword_n = integer()
    ))
  }
  data.table::rbindlist(results, use.names = TRUE, fill = TRUE)
}
