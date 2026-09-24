# MLM top-k prediction inspection: what whole words / subword completions the
# masked language model expected at each position of a sentence, alongside the
# surprisal of what was actually written. Complements fnt_surprisal.R (which
# reports *how surprised* the model was) by exposing *what it expected instead*.

suppressPackageStartupMessages(library(data.table))

# Strip subword-piece markers (WordPiece "##", SentencePiece "▁", byte-level
# BPE "Ġ") for display and top-k membership checks.
.clean_piece <- function(x) {
  return(sub("^##", "", sub("^[▁Ġ]", "", x)))
}

# Load the MLM backend and (re)load the model only when the requested model is
# not already the one held in the Python session, mirroring the reload guard in
# fnt_surprisal.R. Binds the Python scorer functions into globalenv() so the
# wrappers below can call them after this helper returns.
.ensure_mlm_model <- function(model_name) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")
  }
  reticulate::py_require(c("torch", "transformers"), action = "add")
  reticulate::source_python("py/llm_scoring.py", envir = globalenv())
  st <- tryCatch(get_llm_state(), error = function(e) NULL)
  if (is.null(st$model_name) || st$model_name != model_name ||
      !identical(st$mode, "mlm")) {
    load_llm_model(model_name, mode = "mlm")
  }
  return(invisible(TRUE))
}

#' Top-k predictions for every masked position of one sentence
#'
#' Scores a sentence under a masked language model with the same per-subword
#' PLL scorer used by \code{\link{llm_surprisal_entropy}}, and additionally
#' returns, for each subword position, the \code{k} fill-ins the model judged
#' most probable at that mask. High-surprisal tokens can then be read against
#' what the model expected instead of only against a surprisal value.
#'
#' Because sibling subwords of a multi-subword word stay visible under
#' \code{pll_mode = "original"}, the top-k at a piece position are word
#' \emph{completions}, not whole words. For whole-word alternatives use
#' \code{\link{llm_word_predictions}} (single-piece) or
#' \code{\link{llm_beam_word_predictions}} (multi-piece).
#'
#' @param sentence Raw sentence string (not pre-tokenised).
#' @param context Optional priming string prepended to the sentence.
#' @param k Number of predictions to return per position.
#' @param model_name HuggingFace model identifier.
#' @param pll_mode PLL scoring mode: \code{"original"} (Salazar et al. 2020)
#'   or \code{"within_word_l2r"} (Kauf & Ivanova 2023).
#' @param temperature Softmax temperature (default 1.0).
#' @param batch_size Batch size for MLM scoring.
#' @returns A \code{data.table}, one row per subword, with columns
#'   \code{word_id}, \code{word} (as the tokenizer sees it), \code{subword}
#'   (the masked piece), \code{surprisal} (bits), \code{entropy} (bits),
#'   \code{actual_in_topk} (logical), and \code{top_predictions}
#'   ("mot (.42) | mot2 (.13) | ..."). Special-token rows are dropped.
#' @references
#'   Salazar, J., Liang, D., Nguyen, T. Q., & Kirchhoff, K. (2020). Masked
#'   language model scoring. In \emph{Proceedings of ACL} (pp. 2699--2712).
#'   \doi{10.18653/v1/2020.acl-main.240}
#' @citation_type "adapted" (PLL scorer of Salazar et al. 2020, extended to
#'   return top-k fill-ins per position)
#' @export
llm_top_predictions <- function(sentence,
                                context = NULL,
                                k = 5L,
                                model_name = "almanach/moderncamembert-base",
                                pll_mode = "original",
                                temperature = 1.0,
                                batch_size = 8L) {
  .ensure_mlm_model(model_name)

  score <- score_masked_lm_tokens(
    tokens              = sentence,
    temperature         = temperature,
    batch_size          = batch_size,
    context_text        = context,
    is_split_into_words = FALSE,
    pll_mode            = pll_mode,
    top_k               = as.integer(k)
  )

  wids <- vapply(score$subword_word_ids,
                 function(w) if (is.null(w)) NA_integer_ else as.integer(w),
                 integer(1))
  words <- unlist(score$word_tokens)
  pieces <- unlist(score$subword_tokens)
  surps <- unlist(score$subword_surprisals)
  ents <- unlist(score$subword_entropies)

  pred_str <- mapply(function(toks, prbs) {
    if (length(toks) == 0) return("")
    paste(sprintf("%s (%.2f)", unlist(toks), unlist(prbs)), collapse = " | ")
  }, score$subword_topk_tokens, score$subword_topk_probs)

  actual_in_topk <- mapply(function(piece, toks) {
    if (length(toks) == 0) return(NA)
    .clean_piece(piece) %in% unlist(toks)
  }, pieces, score$subword_topk_tokens)

  dt <- data.table(
    word_id         = wids + 1L,
    word            = ifelse(is.na(wids), "", words[wids + 1L]),
    subword         = pieces,
    surprisal       = surps,
    entropy         = ents,
    actual_in_topk  = actual_in_topk,
    top_predictions = pred_str
  )
  return(dt[!is.na(word_id)])
}

#' Whole-word alternatives for every word of a sentence (single-mask)
#'
#' Masks each word's entire subword span with a single \code{[MASK]} and reads
#' the top-k of that distribution, so predictions are complete words rather
#' than word pieces. Also runs the per-piece PLL scorer so each word carries
#' the same \code{surprisal} as \code{\link{llm_surprisal_entropy}}.
#'
#' Caveat: a single mask is filled by a single vocabulary token, so candidate
#' words the tokenizer would split into several pieces cannot appear among the
#' predictions. \code{word_prob} (probability of the written word in the
#' whole-word distribution) is \code{NA} for multi-piece words for the same
#' reason. Use \code{\link{llm_beam_word_predictions}} to recover multi-piece
#' candidates.
#'
#' @param sentence Raw sentence string (not pre-tokenised).
#' @param context Optional priming string prepended to the sentence.
#' @param k Number of predictions to return per word.
#' @param model_name HuggingFace model identifier.
#' @param pll_mode PLL scoring mode for the accompanying surprisal:
#'   \code{"original"} or \code{"within_word_l2r"}.
#' @param temperature Softmax temperature (default 1.0).
#' @param batch_size Batch size for MLM scoring.
#' @returns A \code{data.table}, one row per word, with columns \code{word_id},
#'   \code{word}, \code{surprisal} (bits, PLL), \code{n_pieces},
#'   \code{word_prob} (whole-word probability, \code{NA} for multi-piece
#'   words), \code{actual_in_topk} (logical), and \code{top_words}.
#' @references
#'   Salazar, J., Liang, D., Nguyen, T. Q., & Kirchhoff, K. (2020). Masked
#'   language model scoring. In \emph{Proceedings of ACL} (pp. 2699--2712).
#'   \doi{10.18653/v1/2020.acl-main.240}
#' @citation_type "adapted" (whole-word masking on top of the Salazar et al.
#'   2020 PLL scorer)
#' @export
llm_word_predictions <- function(sentence,
                                 context = NULL,
                                 k = 5L,
                                 model_name = "almanach/moderncamembert-base",
                                 pll_mode = "original",
                                 temperature = 1.0,
                                 batch_size = 8L) {
  .ensure_mlm_model(model_name)

  wp <- topk_whole_word_predictions(
    tokens              = sentence,
    top_k               = as.integer(k),
    temperature         = temperature,
    batch_size          = batch_size,
    context_text        = context,
    is_split_into_words = FALSE
  )
  sc <- score_masked_lm_tokens(
    tokens              = sentence,
    temperature         = temperature,
    batch_size          = batch_size,
    context_text        = context,
    is_split_into_words = FALSE,
    pll_mode            = pll_mode
  )

  words <- unlist(wp$word_tokens)
  n <- length(words)

  top_words <- mapply(function(toks, prbs) {
    if (length(toks) == 0) return("")
    paste(sprintf("%s (%.2f)", unlist(toks), unlist(prbs)), collapse = " | ")
  }, wp$word_topk_tokens, wp$word_topk_probs)

  actual_in_topk <- mapply(function(w, toks) {
    if (length(toks) == 0) return(NA)
    w %in% unlist(toks)
  }, words, wp$word_topk_tokens)

  word_prob <- unlist(wp$word_actual_prob)
  word_prob[is.nan(word_prob)] <- NA_real_

  dt <- data.table(
    word_id        = seq_len(n),
    word           = words,
    surprisal      = unlist(sc$word_surprisals),
    n_pieces       = as.integer(unlist(wp$word_n_pieces)),
    word_prob      = word_prob,
    actual_in_topk = actual_in_topk,
    top_words      = top_words
  )
  return(dt)
}

#' Whole-word alternatives including multi-piece words (beam search)
#'
#' Like \code{\link{llm_word_predictions}} but the masked span is refilled with
#' 1..\code{max_pieces} masks and completed left-to-right by beam search, so
#' candidates the tokenizer splits into several pieces (e.g. "fumaient",
#' "chassé") can appear. Probabilities are joint (chained) piece probabilities;
#' single-piece candidates match the single-mask view. \code{word_prob} is
#' defined for every word here (same chaining).
#'
#' Slower than the single-mask view: roughly
#' \code{sum(1..max_pieces) * beam_width} forward passes per word.
#'
#' @param sentence Raw sentence string (not pre-tokenised).
#' @param context Optional priming string prepended to the sentence.
#' @param k Number of predictions to return per word.
#' @param beam_width Beam width for the left-to-right piece search.
#' @param max_pieces Maximum number of subword pieces a candidate word may
#'   have.
#' @param model_name HuggingFace model identifier.
#' @param pll_mode PLL scoring mode for the accompanying surprisal:
#'   \code{"original"} or \code{"within_word_l2r"}.
#' @param temperature Softmax temperature (default 1.0).
#' @param batch_size Batch size for MLM scoring.
#' @returns A \code{data.table}, one row per word, with columns \code{word_id},
#'   \code{word}, \code{surprisal} (bits, PLL), \code{n_pieces},
#'   \code{word_prob} (joint probability of the written word), \code{
#'   actual_in_topk} (logical), and \code{top_words}.
#' @references
#'   Salazar, J., Liang, D., Nguyen, T. Q., & Kirchhoff, K. (2020). Masked
#'   language model scoring. In \emph{Proceedings of ACL} (pp. 2699--2712).
#'   \doi{10.18653/v1/2020.acl-main.240}
#' @citation_type "adapted" (multi-piece beam completion on top of the Salazar
#'   et al. 2020 PLL scorer)
#' @export
llm_beam_word_predictions <- function(sentence,
                                      context = NULL,
                                      k = 5L,
                                      beam_width = 5L,
                                      max_pieces = 3L,
                                      model_name = "almanach/moderncamembert-base",
                                      pll_mode = "original",
                                      temperature = 1.0,
                                      batch_size = 8L) {
  .ensure_mlm_model(model_name)

  wp <- beam_word_predictions(
    tokens              = sentence,
    top_k               = as.integer(k),
    beam_width          = as.integer(beam_width),
    max_pieces          = as.integer(max_pieces),
    temperature         = temperature,
    context_text        = context,
    is_split_into_words = FALSE
  )
  sc <- score_masked_lm_tokens(
    tokens              = sentence,
    temperature         = temperature,
    batch_size          = batch_size,
    context_text        = context,
    is_split_into_words = FALSE,
    pll_mode            = pll_mode
  )

  words <- unlist(wp$word_tokens)
  n <- length(words)

  top_words <- mapply(function(toks, prbs) {
    if (length(toks) == 0) return("")
    paste(sprintf("%s (%.2f)", unlist(toks), unlist(prbs)), collapse = " | ")
  }, wp$word_topk_tokens, wp$word_topk_probs)

  actual_in_topk <- mapply(function(w, toks) {
    if (length(toks) == 0) return(NA)
    w %in% unlist(toks)
  }, words, wp$word_topk_tokens)

  word_prob <- unlist(wp$word_actual_prob)
  word_prob[is.nan(word_prob)] <- NA_real_

  dt <- data.table(
    word_id        = seq_len(n),
    word           = words,
    surprisal      = unlist(sc$word_surprisals),
    n_pieces       = as.integer(unlist(wp$word_n_pieces)),
    word_prob      = word_prob,
    actual_in_topk = actual_in_topk,
    top_words      = top_words
  )
  return(dt)
}

# Pad by character count (sprintf "%-Ns" pads by bytes, so accented words
# would misalign in a fixed-width table).
.pad_chars <- function(x, w) {
  x <- substr(x, 1, w)
  return(paste0(x, strrep(" ", pmax(0, w - nchar(x)))))
}

#' Pretty-print a whole-word prediction table
#'
#' One line per word for the output of \code{\link{llm_word_predictions}} or
#' \code{\link{llm_beam_word_predictions}}: PLL surprisal (bits), the
#' whole-word probability of what was written ("-" when \code{NA}), a "*" when
#' the written word is among the top-k, then the model's predictions.
#'
#' @param dt A \code{data.table} from \code{\link{llm_word_predictions}} or
#'   \code{\link{llm_beam_word_predictions}}.
#' @returns \code{dt}, invisibly. Called for its printed side effect.
#' @export
print_word_predictions <- function(dt) {
  cat(.pad_chars("word", 18), "  surp  p(word)    top predicted words\n",
      sep = "")
  cat(strrep("-", 95), "\n", sep = "")
  cat(sprintf("%s %6.1f  %s %-2s %s\n",
              .pad_chars(dt$word, 18),
              dt$surprisal,
              ifelse(is.na(dt$word_prob), "   -  ",
                     sprintf("%6.2f", dt$word_prob)),
              ifelse(!is.na(dt$actual_in_topk) & dt$actual_in_topk, "*", ""),
              dt$top_words),
      sep = "")
  return(invisible(dt))
}

#' Pretty-print a subword prediction table
#'
#' One line per subword for the output of \code{\link{llm_top_predictions}}:
#' the word written, its surprisal in bits, a "*" when the written form is
#' among the top-k, then the predictions. Multi-subword words repeat the word
#' with the masked piece in brackets.
#'
#' @param dt A \code{data.table} from \code{\link{llm_top_predictions}}.
#' @returns \code{dt}, invisibly. Called for its printed side effect.
#' @export
print_top_predictions <- function(dt) {
  label <- ifelse(duplicated(dt$word_id),
                  sprintf("%s [%s]", dt$word, .clean_piece(dt$subword)),
                  dt$word)
  cat(.pad_chars("word", 18), "  surp    top predictions\n", sep = "")
  cat(strrep("-", 90), "\n", sep = "")
  cat(sprintf("%s %6.1f %-2s %s\n",
              .pad_chars(label, 18),
              dt$surprisal,
              ifelse(!is.na(dt$actual_in_topk) & dt$actual_in_topk, "*", ""),
              dt$top_predictions),
      sep = "")
  return(invisible(dt))
}
