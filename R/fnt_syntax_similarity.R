# Functions for syntactic similarity between dependency trees.
#
# Uses Tree Edit Distance (APTED algorithm, Pawlik & Augsten 2015) to measure
# how structurally different two dependency trees are.  Requires the 'apted'
# Python package accessible via reticulate.
#
# Two document-level features follow FABRA nomenclature:
#   syn_local_sim  — mean similarity between adjacent sentence pairs
#   syn_global_sim — mean similarity across all sentence pairs in a document
#
# loignon.guillaume@uqam.ca


# --- Built-in prototype -------------------------------------------------------

# Pre-parsed prototype sentences (French GSD model + post_process_lexicon).
# Used as defaults for sentence_prototype_ted() — no UDPipe model load required.
#
# .PROTOTYPE_FR_SV  — "Il mange."              minimal subject-verb (2 content nodes)
# .PROTOTYPE_FR_SVO — "Le chat mange le poisson." canonical SVO     (5 content nodes)

.PROTOTYPE_FR_SV <- data.table::data.table(
  doc_id        = ".prototype",
  sentence_id   = 1L,
  token_id      = c("1", "2", "3"),
  head_token_id = c("2", "0", "2"),
  token         = c("il", "mange", "."),
  lemma         = c("il", "manger", "."),
  upos          = c("PRON", "VERB", "PUNCT"),
  dep_rel       = c("nsubj", "root", "punct"),
  compte        = c(TRUE, TRUE, FALSE)
)

.PROTOTYPE_FR_SVO <- data.table::data.table(
  doc_id        = ".prototype",
  sentence_id   = 1L,
  token_id      = c("1", "2", "3", "4", "5", "6"),
  head_token_id = c("2", "3", "0", "5", "3", "3"),
  token         = c("le", "chat", "mange", "le", "poisson", "."),
  lemma         = c("le", "chat", "manger", "le", "poisson", "."),
  upos          = c("DET", "NOUN", "VERB", "DET", "NOUN", "PUNCT"),
  dep_rel       = c("det", "nsubj", "root", "det", "obj", "punct"),
  compte        = c(TRUE, TRUE, TRUE, TRUE, TRUE, FALSE)
)


# --- Internal helpers ---------------------------------------------------------

#' Load the Python TED backend (once per session)
#'
#' @keywords internal
.load_ted_backend <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install with install.packages('reticulate').")
  }
  reticulate::py_require("apted", action = "add")
  reticulate::source_python(here::here("py/ted_backend.py"))
}

#' Extract label vector from a sentence data.table
#'
#' @param dt_sent   data.table for one sentence.
#' @param label_mode Character; \code{"upos"}, \code{"dep_rel"}, or
#'   \code{"combined"} (upos and dep_rel joined by \code{"_"}).
#' @returns Character vector of node labels, one per token.
#' @keywords internal
.sent_labels <- function(dt_sent, label_mode) {
  switch(label_mode,
    upos     = dt_sent$upos,
    dep_rel  = dt_sent$dep_rel,
    combined = paste(dt_sent$upos, dt_sent$dep_rel, sep = "_"),
    stop("label_mode must be one of: 'upos', 'dep_rel', 'combined'")
  )
}

#' Build a reticulate-ready list for one sentence
#'
#' @keywords internal
.sent_to_py_args <- function(dt_sent, label_mode) {
  list(
    token_ids = as.character(dt_sent$token_id),
    head_ids  = as.character(dt_sent$head_token_id),
    labels    = .sent_labels(dt_sent, label_mode)
  )
}


# --- Public functions ---------------------------------------------------------

#' Compute Tree Edit Distance between sentence pairs
#'
#' Returns one row per sentence pair (adjacent pairs only, or all pairs).
#' TED is computed via the APTED algorithm on the dependency trees produced
#' by \code{post_process_lexicon()}.
#'
#' A normalized TED of 0 means the two trees are identical; 1 means maximally
#' different.  Similarity is \code{1 - normalized_ted}.
#'
#' @param dt_corpus   data.table from \code{post_process_lexicon()}.
#' @param label_mode  Character; node label used for the cost function.
#'   \code{"upos"} (default), \code{"dep_rel"}, or \code{"combined"}.
#' @param normalize   Logical; if \code{TRUE} (default), divide raw TED by
#'   \code{max(size_a, size_b) + 1} to obtain a value in \code{[0, 1]}.
#' @param adjacent_only Logical; if \code{TRUE} (default), only compute TED
#'   between consecutive sentences.  If \code{FALSE}, compute all pairs.
#' @param include_punct Logical; if \code{FALSE} (default), exclude PUNCT
#'   tokens from the trees before computing TED.
#' @returns A \code{data.table} with columns:
#'   \code{doc_id}, \code{sent_id_a}, \code{sent_id_b},
#'   \code{ted}, \code{n_tokens_a}, \code{n_tokens_b}.
#'
#' @references
#'   Pawlik, M., & Augsten, N. (2015). Efficient computation of the tree edit
#'   distance. *ACM Transactions on Database Systems*, 40(1), 1--40.
#'   \doi{10.1145/2699485}
#'
#'   Zhang, K., & Shasha, D. (1989). Simple fast algorithms for the editing
#'   distance between trees and related problems. *SIAM Journal on Computing*,
#'   18(6), 1245--1262. \doi{10.1137/0218082}
#'
#' @export
sentence_ted_pairs <- function(dt_corpus,
                               label_mode    = "upos",
                               normalize     = TRUE,
                               adjacent_only = TRUE,
                               include_punct = FALSE) {
  .load_ted_backend()

  dt <- data.table::as.data.table(dt_corpus)
  if (!include_punct) {
    dt <- dt[upos != "PUNCT"]
  }

  all_docs <- unique(dt$doc_id)
  rows <- vector("list", length(all_docs))

  for (d_idx in seq_along(all_docs)) {
    doc <- all_docs[d_idx]
    dt_doc <- dt[doc_id == doc]
    sent_ids <- sort(unique(dt_doc$sentence_id))
    n_sents  <- length(sent_ids)

    if (n_sents < 2L) next

    if (adjacent_only) {
      pairs <- data.frame(a = sent_ids[-n_sents], b = sent_ids[-1L])
    } else {
      idx   <- which(upper.tri(matrix(0L, n_sents, n_sents)), arr.ind = TRUE)
      pairs <- data.frame(a = sent_ids[idx[, 1L]], b = sent_ids[idx[, 2L]])
    }

    n_pairs   <- nrow(pairs)
    ted_vals  <- numeric(n_pairs)
    n_toks_a  <- integer(n_pairs)
    n_toks_b  <- integer(n_pairs)

    for (p in seq_len(n_pairs)) {
      sa <- dt_doc[sentence_id == pairs$a[p]]
      sb <- dt_doc[sentence_id == pairs$b[p]]

      args_a <- .sent_to_py_args(sa, label_mode)
      args_b <- .sent_to_py_args(sb, label_mode)

      ted_vals[p] <- reticulate::py$compute_ted(
        args_a$token_ids, args_a$head_ids, args_a$labels,
        args_b$token_ids, args_b$head_ids, args_b$labels,
        normalize = normalize
      )
      n_toks_a[p] <- nrow(sa)
      n_toks_b[p] <- nrow(sb)
    }

    rows[[d_idx]] <- data.table::data.table(
      doc_id     = doc,
      sent_id_a  = pairs$a,
      sent_id_b  = pairs$b,
      ted        = ted_vals,
      n_tokens_a = n_toks_a,
      n_tokens_b = n_toks_b
    )
  }

  result <- data.table::rbindlist(rows)
  if (nrow(result) == 0L) {
    return(data.table::data.table(
      doc_id     = character(),
      sent_id_a  = integer(),
      sent_id_b  = integer(),
      ted        = numeric(),
      n_tokens_a = integer(),
      n_tokens_b = integer()
    ))
  }
  data.table::setkey(result, doc_id, sent_id_a, sent_id_b)
  return(result)
}


#' Document-level syntactic similarity
#'
#' Computes global and local syntactic similarity scores per document, following
#' FABRA's \code{SYNdevSIM} (global) and \code{SYNdevSIMA} (local) features.
#'
#' Both scores are similarity values in \code{[0, 1]}: \code{1 - normalized_TED}.
#' Higher = more similar (more uniform syntactic structure across sentences).
#'
#' @param dt_corpus   data.table from \code{post_process_lexicon()}.
#' @param label_mode  Character; \code{"upos"} (default), \code{"dep_rel"},
#'   or \code{"combined"}.
#' @param include_punct Logical; exclude PUNCT tokens (default \code{FALSE}).
#' @returns A \code{data.table} keyed by \code{doc_id} with columns:
#'   \code{syn_local_sim}, \code{syn_global_sim},
#'   \code{syn_local_ted}, \code{syn_global_ted},
#'   \code{n_sent_pairs_local}, \code{n_sent_pairs_global}.
#'
#' @references
#'   Crossley, S. A., Kyle, K., & McNamara, D. S. (2016). The tool for the
#'   automatic analysis of text cohesion (TAACO): Automatic assessment of
#'   local, global, and text cohesion. *Behavior Research Methods*, 48(4),
#'   1227--1237. \doi{10.3758/s13428-015-0651-7}
#'
#'   Pawlik, M., & Augsten, N. (2015). Efficient computation of the tree edit
#'   distance. *ACM Transactions on Database Systems*, 40(1), 1--40.
#'   \doi{10.1145/2699485}
#'
#' @export
doc_syntax_similarity <- function(dt_corpus,
                                  label_mode    = "upos",
                                  include_punct = FALSE) {
  dt_local  <- sentence_ted_pairs(dt_corpus,
                                  label_mode    = label_mode,
                                  normalize     = TRUE,
                                  adjacent_only = TRUE,
                                  include_punct = include_punct)

  dt_global <- sentence_ted_pairs(dt_corpus,
                                  label_mode    = label_mode,
                                  normalize     = TRUE,
                                  adjacent_only = FALSE,
                                  include_punct = include_punct)

  agg_local <- dt_local[, .(
    syn_local_ted      = mean(ted),
    syn_local_sim      = 1 - mean(ted),
    n_sent_pairs_local = .N
  ), by = doc_id]

  agg_global <- dt_global[, .(
    syn_global_ted      = mean(ted),
    syn_global_sim      = 1 - mean(ted),
    n_sent_pairs_global = .N
  ), by = doc_id]

  result <- merge(agg_local, agg_global, by = "doc_id", all = TRUE)
  data.table::setkey(result, doc_id)
  return(result)
}


#' Compute TED between each corpus sentence and a prototype sentence
#'
#' Parses \code{prototype} once, then returns the Tree Edit Distance between
#' every sentence in \code{dt_corpus} and that prototype tree.  Lower distance
#' means the sentence is more structurally similar to the prototype.
#'
#' The prototype is parsed with the same UDPipe model used for the corpus.
#' Pass a pre-parsed prototype via \code{dt_prototype} to skip re-parsing
#' (useful when calling this function repeatedly in a loop).
#'
#' @param dt_corpus   data.table from \code{post_process_lexicon()}.
#' @param prototype   Character string; the prototype sentence to parse and
#'   compare against.  Ignored when \code{dt_prototype} is supplied.
#' @param ud_model    Path to the UDPipe model file used to parse the
#'   prototype (default: \code{"models/french_gsd-remix_3.udpipe"}).
#' @param dt_prototype Optional pre-parsed prototype data.table (output of
#'   \code{post_process_lexicon()}) for a single-sentence corpus.  When
#'   supplied, \code{prototype} and \code{ud_model} are ignored.
#' @param label_mode  Character; node label used for the cost function.
#'   \code{"upos"} (default), \code{"dep_rel"}, or \code{"combined"}.
#' @param normalize   Logical; if \code{TRUE} (default), divide raw TED by
#'   \code{max(size_corpus_sent, size_prototype) + 1}.
#' @param include_punct Logical; exclude PUNCT tokens (default \code{FALSE}).
#' @returns A \code{data.table} keyed by \code{doc_id}, \code{sentence_id}
#'   with columns \code{prototype_ted} (distance) and \code{prototype_sim}
#'   (1 - normalized distance).
#'
#' @references
#'   Pawlik, M., & Augsten, N. (2015). Efficient computation of the tree edit
#'   distance. *ACM Transactions on Database Systems*, 40(1), 1--40.
#'   \doi{10.1145/2699485}
#'
#' @export
sentence_prototype_ted <- function(dt_corpus,
                                   prototype     = NULL,
                                   ud_model      = "models/french_gsd-remix_3.udpipe",
                                   dt_prototype  = .PROTOTYPE_FR_SVO,
                                   label_mode    = "upos",
                                   normalize     = TRUE,
                                   include_punct = FALSE) {
  .load_ted_backend()

  if (!is.null(prototype)) {
    # Parse a custom prototype sentence on the fly
    if (!exists("parse_text", mode = "function")) {
      stop("parse_text() not found. Source R/fnt_corpus.R before passing a custom prototype.")
    }
    dt_proto_raw <- parse_text(
      data.frame(doc_id = ".prototype", text = prototype),
      ud_model      = ud_model,
      show_progress = FALSE
    )
    dt_prototype <- post_process_lexicon(dt_proto_raw)
  }

  dt_proto_sent <- dt_prototype[sentence_id == min(dt_prototype$sentence_id)]
  if (!include_punct) {
    dt_proto_sent <- dt_proto_sent[upos != "PUNCT"]
  }
  proto_args <- .sent_to_py_args(dt_proto_sent, label_mode)

  dt <- data.table::as.data.table(dt_corpus)
  if (!include_punct) {
    dt <- dt[upos != "PUNCT"]
  }

  sent_keys <- unique(dt[, .(doc_id, sentence_id)])
  n <- nrow(sent_keys)
  ted_vals <- numeric(n)

  for (i in seq_len(n)) {
    ds <- dt[doc_id == sent_keys$doc_id[i] & sentence_id == sent_keys$sentence_id[i]]
    corp_args <- .sent_to_py_args(ds, label_mode)
    ted_vals[i] <- reticulate::py$compute_ted(
      corp_args$token_ids, corp_args$head_ids, corp_args$labels,
      proto_args$token_ids, proto_args$head_ids, proto_args$labels,
      normalize = normalize
    )
  }

  result <- data.table::data.table(
    doc_id        = sent_keys$doc_id,
    sentence_id   = sent_keys$sentence_id,
    prototype_ted = ted_vals,
    prototype_sim = 1 - ted_vals
  )
  data.table::setkey(result, doc_id, sentence_id)
  return(result)
}


#' Find corpus sentences structurally similar to a prototype
#'
#' Parses \code{prototype}, scores every sentence in \code{dt_corpus} by Tree
#' Edit Distance, and returns the \code{n} closest matches with their surface
#' text reconstructed from tokens.
#'
#' Useful for structure-based corpus search: supply any target construction
#' (relative clause, passive, cleft, etc.) as \code{prototype} and retrieve
#' authentic corpus examples ranked by structural proximity.
#'
#' @param dt_corpus    data.table from \code{post_process_lexicon()}.
#' @param prototype    Character string; the target sentence structure to
#'   search for.  Parsed on the fly — requires \code{parse_text()} and
#'   \code{post_process_lexicon()} to be available (i.e. source
#'   \code{R/fnt_corpus.R} first).  Pass \code{dt_prototype} instead to
#'   skip parsing.
#' @param ud_model     Path to the UDPipe model (default French GSD).
#' @param dt_prototype Optional pre-parsed prototype data.table.
#' @param label_mode   \code{"upos"} (default), \code{"dep_rel"}, or
#'   \code{"combined"}.
#' @param n            Number of closest sentences to return (default 10).
#' @param include_punct Logical; exclude PUNCT tokens (default \code{FALSE}).
#' @returns A \code{data.table} with columns \code{doc_id},
#'   \code{sentence_id}, \code{prototype_sim}, \code{prototype_ted},
#'   and \code{sentence} (reconstructed surface text), sorted by ascending
#'   TED (most similar first).
#'
#' @references
#'   Pawlik, M., & Augsten, N. (2015). Efficient computation of the tree edit
#'   distance. *ACM Transactions on Database Systems*, 40(1), 1--40.
#'   \doi{10.1145/2699485}
#'
#' @export
find_sentences_by_structure <- function(dt_corpus,
                                        prototype     = NULL,
                                        ud_model      = "models/french_gsd-remix_3.udpipe",
                                        dt_prototype  = .PROTOTYPE_FR_SVO,
                                        label_mode    = "upos",
                                        n             = 10L,
                                        include_punct = FALSE) {
  dt_scores <- sentence_prototype_ted(
    dt_corpus     = dt_corpus,
    prototype     = prototype,
    ud_model      = ud_model,
    dt_prototype  = dt_prototype,
    label_mode    = label_mode,
    normalize     = TRUE,
    include_punct = include_punct
  )

  # Reconstruct surface sentence from all tokens (including PUNCT for readability)
  dt_text <- data.table::as.data.table(dt_corpus)[
    ,
    .(sentence = paste(token, collapse = " ")),
    by = .(doc_id, sentence_id)
  ]

  result <- merge(dt_scores, dt_text, by = c("doc_id", "sentence_id"))
  data.table::setorder(result, prototype_ted)
  return(result[seq_len(min(n, nrow(result))),
                .(doc_id, sentence_id, prototype_sim, prototype_ted, sentence)])
}


#' Extract corpus sentences matching a prototype structure
#'
#' Scores every sentence against the prototype tree and returns all sentences
#' whose normalized TED is at or below \code{threshold}.  Results are printed
#' to the console and optionally written to a TSV file.
#'
#' At the default \code{threshold = 0} the function returns only sentences
#' with a structurally identical tree (same UPOS sequence and attachment
#' pattern).  Raise the threshold to retrieve near-matches.
#'
#' @param dt_corpus    data.table from \code{post_process_lexicon()}.
#' @param prototype    Character string; the target sentence to parse and use
#'   as a structural query.  Requires \code{parse_text()} and
#'   \code{post_process_lexicon()} to be available.  Ignored when
#'   \code{dt_prototype} is supplied.
#' @param ud_model     Path to the UDPipe model (default French GSD).
#' @param dt_prototype Optional pre-parsed prototype data.table.  Defaults to
#'   the built-in SVO prototype (\code{.PROTOTYPE_FR_SVO}).
#' @param label_mode   \code{"upos"} (default), \code{"dep_rel"}, or
#'   \code{"combined"}.
#' @param threshold    Maximum normalized TED to include.  \code{0} (default)
#'   returns structurally identical sentences only; \code{0.2} includes close
#'   variants; \code{1} returns all sentences.
#' @param n            Maximum number of results to print and return.  Default
#'   \code{10}; use \code{Inf} to return all matches.
#' @param include_punct Logical; exclude PUNCT tokens from the trees
#'   (default \code{FALSE}).
#' @param output_file  Optional file path.  When supplied, results are written
#'   as a tab-separated file with columns \code{doc_id}, \code{sentence_id},
#'   \code{prototype_ted}, \code{prototype_sim}, and \code{sentence}.
#' @returns Invisibly, a \code{data.table} of matching sentences sorted by
#'   ascending TED then \code{doc_id} and \code{sentence_id}.
#'
#' @references
#'   Pawlik, M., & Augsten, N. (2015). Efficient computation of the tree edit
#'   distance. *ACM Transactions on Database Systems*, 40(1), 1--40.
#'   \doi{10.1145/2699485}
#'
#' @export
corpus_search_by_structure <- function(dt_corpus,
                                       prototype     = NULL,
                                       ud_model      = "models/french_gsd-remix_3.udpipe",
                                       dt_prototype  = .PROTOTYPE_FR_SVO,
                                       label_mode    = "upos",
                                       threshold     = 0,
                                       n             = 10L,
                                       include_punct = FALSE,
                                       output_file   = NULL) {
  dt_scores <- sentence_prototype_ted(
    dt_corpus     = dt_corpus,
    prototype     = prototype,
    ud_model      = ud_model,
    dt_prototype  = dt_prototype,
    label_mode    = label_mode,
    normalize     = TRUE,
    include_punct = include_punct
  )

  dt_text <- data.table::as.data.table(dt_corpus)[
    ,
    .(sentence = paste(token, collapse = " ")),
    by = .(doc_id, sentence_id)
  ]

  result <- merge(dt_scores, dt_text, by = c("doc_id", "sentence_id"))
  result <- result[prototype_ted <= threshold + 1e-9]
  data.table::setorder(result, prototype_ted, doc_id, sentence_id)
  result <- result[, .(doc_id, sentence_id, prototype_ted, prototype_sim, sentence)]

  n_found   <- nrow(result)
  n_display <- if (is.infinite(n)) n_found else min(n, n_found)
  message(sprintf(
    "%d sentence(s) with TED <= %.2f (similarity >= %.2f)%s:",
    n_found, threshold, 1 - threshold,
    if (n_display < n_found) sprintf(" — showing top %d", n_display) else ""
  ))
  if (n_found == 0L) {
    message("  (none found — try raising the threshold)")
  } else {
    for (i in seq_len(n_display)) {
      cat(sprintf("  [%s s%d | sim=%.2f] %s\n",
                  result$doc_id[i], result$sentence_id[i],
                  result$prototype_sim[i], result$sentence[i]))
    }
  }

  result <- result[seq_len(n_display)]

  if (!is.null(output_file)) {
    data.table::fwrite(result, output_file, sep = "\t")
    message("Written to ", output_file)
  }

  return(invisible(result))
}
