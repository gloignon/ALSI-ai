library(data.table)

#' Score neural (full-context) POS surprisal and entropy on a parsed corpus
#'
#' A neural counterpart to \code{pos_surprisal()}. Instead of a fixed-order
#' trigram, surprisal and entropy are computed from a small GRU language model
#' over UPOS tag sequences, conditioned on the \emph{entire} left context of the
#' sentence:
#' \deqn{surprisal(t_i) = -\log_2 P(t_i \mid \langle bos\rangle, t_1, ..., t_{i-1})}
#' \deqn{entropy(t_i)   = H(P(\cdot \mid \langle bos\rangle, t_1, ..., t_{i-1}))}
#' both in bits. Because the model sees a \code{<bos>} marker, every token is
#' scored — including the first one, which the trigram model cannot score
#' without explicit sentence boundaries.
#'
#' The model is the ALSI verb-modified scheme (copular \code{etre} tagged
#' \code{VERB}, not \code{AUX}), trained on
#' \code{fr_gsd-ud-train_alsi-verbpos.conllu}. Score a corpus parsed with the
#' matching ALSI UDPipe model.
#'
#' Output columns are suffixed \code{_nn} so they sit alongside the trigram
#' \code{pos_surprisal()} columns in the same feature table for direct
#' comparison.
#'
#' @param dt_corpus data.table with at minimum \code{doc_id},
#'   \code{sentence_id}, \code{token_id}, \code{upos}. Rows must be in surface
#'   order within each sentence.
#' @param model_path Path to the exported checkpoint. Default
#'   \code{"models/pos_lm_fr_gsd_alsi.pt"}.
#' @param exclude_pos Character vector of UPOS tags to drop before forming tag
#'   sequences (so they do not interrupt the context). Mirror the choice used
#'   for \code{pos_surprisal()} when comparing the two.
#' @returns A named list of three data.tables:
#'   \describe{
#'     \item{\code{token_surprisal}}{\code{doc_id}, \code{sentence_id},
#'       \code{token_id}, \code{upos}, \code{pos_surprisal_nn},
#'       \code{pos_entropy_nn}, \code{pos_entropy_reduction_nn}.}
#'     \item{\code{sent_surprisal}}{Per-sentence mean/sd of the three measures.}
#'     \item{\code{doc_surprisal}}{Per-document mean/sd of the three measures.}
#'   }
pos_surprisal_nn <- function(dt_corpus,
                             model_path = "models/pos_lm_fr_gsd_alsi.pt",
                             exclude_pos = character(0)) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install it with install.packages('reticulate').")
  }
  reticulate::py_require("torch", action = "add")
  reticulate::source_python("py/pos_lm_scoring.py")

  dt <- setDT(copy(dt_corpus))[, .(doc_id, sentence_id, token_id, upos)]
  dt[, .ord := .I]  # preserve global surface order

  # Drop multi-word-token range rows (e.g. "des" spanning "de"+"les"): UD
  # gives them no upos of their own. Left in, the GRU would encode the
  # missing tag as <pad> and carry that corrupted context through the rest
  # of the sentence's hidden state.
  dt <- dt[!is.na(upos)]

  if (length(exclude_pos) > 0L) {
    dt <- dt[!upos %chin% exclude_pos]
  }

  # One tag sequence per sentence, in surface order. by= keeps groups in order
  # of first appearance; order(.ord) guarantees within-group order.
  sents <- dt[order(.ord), .(tags = list(upos)), by = .(doc_id, sentence_id)]

  scored <- score_pos_sentences(sents$tags, model_path)

  # Flatten back; results align position-by-position with dt[order(.ord)].
  surpr <- unlist(lapply(scored, `[[`, "surprisal"), use.names = FALSE)
  ent   <- unlist(lapply(scored, `[[`, "entropy"),   use.names = FALSE)

  dt <- dt[order(.ord)]
  dt[, pos_surprisal_nn := surpr]
  dt[, pos_entropy_nn   := ent]
  # Entropy reduction: drop in predictive uncertainty from the previous token's
  # context to this one's (positive = uncertainty fell). Mirrors pos_surprisal().
  dt[, pos_entropy_reduction_nn := shift(pos_entropy_nn, 1L) - pos_entropy_nn,
     by = .(doc_id, sentence_id)]

  token_dt <- dt[, .(doc_id, sentence_id, token_id, upos,
                     pos_surprisal_nn, pos_entropy_nn, pos_entropy_reduction_nn)]

  agg <- function(by_cols) {
    token_dt[, .(
      mean_pos_surprisal_nn         = mean(pos_surprisal_nn,         na.rm = TRUE),
      sd_pos_surprisal_nn           = sd(pos_surprisal_nn,           na.rm = TRUE),
      mean_pos_entropy_nn           = mean(pos_entropy_nn,           na.rm = TRUE),
      sd_pos_entropy_nn             = sd(pos_entropy_nn,             na.rm = TRUE),
      mean_pos_entropy_reduction_nn = mean(pos_entropy_reduction_nn, na.rm = TRUE),
      sd_pos_entropy_reduction_nn   = sd(pos_entropy_reduction_nn,   na.rm = TRUE)
    ), by = by_cols]
  }

  list(
    token_surprisal = token_dt,
    sent_surprisal  = agg(c("doc_id", "sentence_id")),
    doc_surprisal   = agg("doc_id")
  )
}
