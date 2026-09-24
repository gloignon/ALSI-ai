
#' Encode sentence embeddings via a Python backend
#'
#' Computes dense vector embeddings for each sentence in a parsed corpus using
#' a HuggingFace model via \pkg{reticulate}. Supports basic, prompt, and query
#' encoding modes.
#'
#' @param dt_corpus A \code{data.table} with \code{doc_id}, \code{sentence_id},
#'   and \code{sentence}.
#' @param model_name HuggingFace model identifier.
#' @param mode Encoding mode: \code{"basic"}, \code{"prompt"}, or \code{"query"}.
#' @param batch_size Batch size for the Python encoder.
#' @param instruction Instruction string for prompt/query modes (ignored in basic).
#' @returns A \code{data.table} with \code{doc_id}, \code{sentence_id},
#'   \code{sentence}, and embedding columns \code{dim1}...\code{dimN}.
encode_embeddings <- function(dt_corpus,
                              model_name = "dangvantuan/french-document-embedding",
                              mode = "basic",
                              batch_size = 32,
                              instruction = "Identifiez le thème principal et les thèmes secondaires dans ce texte.") {
  
  reticulate::py_require(c("sentence-transformers", "torch", "numpy"))
  source_python("py/embed_sentences_instruct.py")

  cat(sprintf("Loading model '%s'...\n", model_name))
  flush.console()
  load_embedding_model(model_name)

  dt_sentences <- as.data.table(dt_corpus)[
    , .(sentence = first(sentence)), by = .(doc_id, sentence_id)
  ][order(doc_id, sentence_id)]
  
  results <- list()
  doc_ids <- unique(dt_sentences$doc_id)
  n_docs <- length(doc_ids)
  total_sentences <- nrow(dt_sentences)
  processed_sentences <- 0
  failed_docs <- character(0)
  
  start_time <- Sys.time()
  
  print_status <- function(processed, total, start_time) {
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    speed <- processed / elapsed * 60  # sentences per minute
    remaining <- (total - processed) / speed * 60  # seconds
    cat(sprintf("\rProcessed %d/%d sentences (%.1f sent/min) — ~%.1f min left     ",
                processed, total, speed, remaining / 60), sep = "")
    flush.console()
  }
  
  for (i in seq_along(doc_ids)) {
    doc <- doc_ids[i]
    doc_dt <- dt_sentences[doc_id == doc]
    n_sent_in_doc <- nrow(doc_dt)
    
    embeddings <- tryCatch({
      emb_matrix <- embed_sentences(doc_dt$sentence, mode = mode,
                                    instruction = instruction, bs = batch_size)

      # reticulate can return 1D for single sentences
      if (is.null(dim(emb_matrix))) {
        emb_matrix <- matrix(emb_matrix, nrow = 1)
      } else {
        emb_matrix <- as.matrix(emb_matrix)
      }

      cbind(doc_dt, as.data.table(emb_matrix))
    }, error = function(e) {
      warning(sprintf("Embedding failed for doc_id = %s: %s", doc, e$message))
      failed_docs <<- c(failed_docs, doc)
      NULL
    })
    
    if (!is.null(embeddings)) {
      results[[length(results) + 1]] <- embeddings
      processed_sentences <- processed_sentences + n_sent_in_doc
    }
    
    if (i %% 3 == 0 || i == n_docs) {
      print_status(processed_sentences, total_sentences, start_time)
    }
  }
  
  dt_embed <- rbindlist(results, use.names = TRUE, fill = TRUE)

  n_dim <- ncol(dt_embed) - 3
  emb_cols <- names(dt_embed)[(ncol(dt_embed) - n_dim + 1):ncol(dt_embed)]
  setnames(dt_embed, emb_cols, paste0("dim", seq_len(n_dim)))
  
  if (length(failed_docs)) {
    message(sprintf("Embedding failed for %d document(s): %s", length(failed_docs),
                    paste(failed_docs, collapse = ", ")))
  }
  
  return(dt_embed)
}


#' Compute sentence and document embeddings for a corpus
#'
#' Wrapper around \code{encode_embeddings} that also computes document-level
#' mean embeddings by averaging sentence vectors.
#'
#' @param dt_corpus A \code{data.table} with \code{doc_id}, \code{sentence_id},
#'   and \code{sentence}.
#' @param model_name HuggingFace model identifier.
#' @param mode Encoding mode: \code{"basic"}, \code{"prompt"}, or \code{"query"}.
#' @param batch_size Batch size for the Python encoder.
#' @param instruction Instruction string for prompt/query modes.
#' @returns A list with \code{dt_sent_embeddings} and \code{dt_doc_embeddings}.
corpus_embeddings <- function(dt_corpus,
                                model_name = "Lajavaness/bilingual-embedding-large",
                                mode = "basic",
                                batch_size = 32,
                                instruction = "Identifiez le thème principal et secondaire dans le texte."
                              ) {

  dt_embeddings <- encode_embeddings(
    dt_corpus,
    model_name = model_name,
    mode = mode,
    batch_size = batch_size,
    instruction = instruction
  )
  
  dim_cols <- grep("^dim", names(dt_embeddings), value = TRUE)
  dt_doc_mean <- dt_embeddings[, lapply(.SD, mean, na.rm = TRUE), by = doc_id, .SDcols = dim_cols]

  return(list(dt_sent_embeddings = dt_embeddings,
              dt_doc_embeddings  = dt_doc_mean))
}


#' Deterministic sentence-pair sample (internal)
#'
#' @param n Number of sentences.
#' @param max_pairs Maximum number of pairs to return.
#' @returns A two-column integer matrix of unique sentence-pair indices.
#' @keywords internal
.sample_sentence_pairs <- function(n, max_pairs = 500L) {
  n_all_pairs <- n * (n - 1L) / 2L
  if (n_all_pairs <= max_pairs) {
    return(which(upper.tri(matrix(FALSE, n, n)), arr.ind = TRUE))
  }

  pair_ids <- floor(seq(0, n_all_pairs - 1L, length.out = max_pairs)) + 1L
  pair_ids <- unique(pair_ids)

  cumulative <- cumsum(seq_len(n - 1L))
  col_pos <- findInterval(pair_ids - 1L, cumulative) + 1L
  prev_cumulative <- c(0L, cumulative)[col_pos]

  cbind(
    row = pair_ids - prev_cumulative,
    col = col_pos + 1L
  )
}

#' Midpoint NN cosine support (internal)
#'
#' Computes the midpoint of each sentence pair, then measures how close that
#' midpoint is to the nearest non-endpoint sentence in the same document.
#'
#' @param mat_norm L2-normalised sentence embedding matrix.
#' @param idx_a Integer vector of first sentence indices.
#' @param idx_b Integer vector of second sentence indices.
#' @returns A numeric vector of nearest-neighbour cosine similarities.
#' @keywords internal
.midpoint_support_scores <- function(mat_norm, idx_a, idx_b) {
  midpoint <- mat_norm[idx_a, , drop = FALSE] + mat_norm[idx_b, , drop = FALSE]
  midpoint_norms <- sqrt(rowSums(midpoint^2))
  midpoint_norms[midpoint_norms == 0] <- 1
  midpoint <- midpoint / midpoint_norms

  sim_midpoint <- midpoint %*% t(mat_norm)
  rows <- seq_along(idx_a)
  sim_midpoint[cbind(rows, idx_a)] <- -Inf
  sim_midpoint[cbind(rows, idx_b)] <- -Inf

  scores <- apply(sim_midpoint, 1, max)
  scores[!is.finite(scores)] <- NA_real_
  scores
}

#' Median within-document support radius (internal)
#'
#' Estimates typical local sentence spacing as the median cosine distance from
#' each sentence to its nearest non-self neighbour.
#'
#' @param mat_norm L2-normalised sentence embedding matrix.
#' @returns A numeric scalar cosine-distance radius.
#' @keywords internal
.sentence_support_radius <- function(mat_norm) {
  sent_sims <- mat_norm %*% t(mat_norm)
  diag(sent_sims) <- -Inf
  nn_sims <- apply(sent_sims, 1, max)
  nn_dist <- 1 - nn_sims
  median(nn_dist[is.finite(nn_dist)], na.rm = TRUE)
}

#' Euclidean k-NN radii for L2-normalised embeddings (internal)
#'
#' @param mat_norm L2-normalised sentence embedding matrix.
#' @param k_neighbors Requested k for local support scale.
#' @returns A list with per-sentence radii and the effective k used.
#' @keywords internal
.sentence_knn_radii <- function(mat_norm, k_neighbors = 10L) {
  n <- nrow(mat_norm)
  effective_k <- min(k_neighbors, n - 1L)
  if (effective_k < 1L) {
    return(list(radii = rep(NA_real_, n), effective_k = 0L))
  }

  d_mat <- as.matrix(dist(mat_norm))
  diag(d_mat) <- Inf
  radii <- apply(d_mat, 1, function(x) sort(x, partial = effective_k)[effective_k])
  list(radii = radii, effective_k = effective_k)
}

#' Local-scale segment support scores (internal)
#'
#' Implements the endpoint-kNN geometric scale from the cluster-convexity spec:
#' q(z) = exp(-0.5 * (d(z) / s(z))^2), with endpoints excluded from the nearest
#' observed sentence search.
#'
#' @param mat_norm L2-normalised sentence embedding matrix.
#' @param idx_a Integer vector of first sentence indices.
#' @param idx_b Integer vector of second sentence indices.
#' @param endpoint_radii Per-sentence k-NN radii.
#' @param occupancy_lambda Multiplier for the binary occupancy threshold.
#' @param scale_epsilon Small lower bound for local scale.
#' @returns A list of continuous support, binary occupancy, and normalised
#'   distances.
#' @keywords internal
.segment_support_scores <- function(mat_norm, idx_a, idx_b, endpoint_radii,
                                    occupancy_lambda = 1,
                                    scale_epsilon = 1e-12) {
  midpoint <- 0.5 * (
    mat_norm[idx_a, , drop = FALSE] +
      mat_norm[idx_b, , drop = FALSE]
  )

  nearest_dist <- numeric(length(idx_a))
  for (i in seq_along(idx_a)) {
    keep <- setdiff(seq_len(nrow(mat_norm)), c(idx_a[i], idx_b[i]))
    if (!length(keep)) {
      nearest_dist[i] <- NA_real_
      next
    }
    deltas <- mat_norm[keep, , drop = FALSE] -
      matrix(midpoint[i, ], nrow = length(keep), ncol = ncol(mat_norm), byrow = TRUE)
    nearest_dist[i] <- min(sqrt(rowSums(deltas^2)))
  }

  local_scale <- sqrt(endpoint_radii[idx_a] * endpoint_radii[idx_b])
  local_scale[!is.finite(local_scale) | local_scale < scale_epsilon] <- scale_epsilon

  norm_dist <- nearest_dist / local_scale
  list(
    support = exp(-0.5 * norm_dist^2),
    occupancy = nearest_dist <= occupancy_lambda * local_scale,
    norm_dist = norm_dist
  )
}

#' Compute document-level coherence features from sentence embeddings
#'
#' Derives a rich set of semantic coherence features per document: thematic
#' dispersion, centroid distance SD, sequential similarity, semantic gaps,
#' topic drift, novelty, estimated number of topics (k-means + silhouette),
#' and embedding space convexity (Gardenfors).
#'
#' @param dt_sent_embeddings A \code{data.table} from \code{corpus_embeddings}
#'   with \code{doc_id}, \code{sentence_id}, and columns \code{dim1}...\code{dimN}.
#' @returns A \code{data.table} with one row per document and \code{emb_*}
#'   columns for each coherence feature.
embedding_coherence <- function(dt_sent_embeddings) {

  dt <- as.data.table(dt_sent_embeddings)
  dim_cols <- grep("^dim", names(dt), value = TRUE)

  if (length(dim_cols) == 0) {
    stop("No embedding columns (dim1, dim2, ...) found in dt_sent_embeddings.")
  }

  doc_ids <- unique(dt$doc_id)
  results <- vector("list", length(doc_ids))

  na_row <- function(doc) {
    data.table(
      doc_id = doc,
      emb_thematic_dispersion = NA_real_,
      emb_centroid_distance_sd = NA_real_,
      emb_sequential_similarity = NA_real_,
      emb_mean_semantic_gap = NA_real_,
      emb_max_semantic_gap = NA_real_,
      emb_topic_drift = NA_real_,
      emb_mean_novelty = NA_real_,
      emb_n_topics = NA_integer_,
      emb_convexity = NA_real_,
      emb_blob_convexity = NA_real_,
      emb_segment_support = NA_real_,
      emb_segment_occupancy = NA_real_,
      emb_local_convexity = NA_real_,
      emb_local_blob_convexity = NA_real_,
      emb_local_segment_support = NA_real_,
      emb_local_segment_occupancy = NA_real_
    )
  }

  for (i in seq_along(doc_ids)) {
    doc <- doc_ids[i]
    mat <- as.matrix(dt[doc_id == doc, ..dim_cols])
    mat <- mat[complete.cases(mat), , drop = FALSE]
    n <- nrow(mat)

    if (n < 2) {
      results[[i]] <- na_row(doc)
      next
    }

    # L2-normalize for cosine similarity via dot product
    norms <- sqrt(rowSums(mat^2))
    norms[norms == 0] <- 1
    mat_norm <- mat / norms

    centroid <- colMeans(mat_norm)
    centroid_norm <- centroid / sqrt(sum(centroid^2))
    dist_to_centroid <- 1 - as.numeric(mat_norm %*% centroid_norm)

    mean_centroid_distance <- mean(dist_to_centroid)
    centroid_dist_sd <- sd(dist_to_centroid)

    consecutive_sims <- rowSums(mat_norm[-n, , drop = FALSE] * mat_norm[-1, , drop = FALSE])
    mean_sequential_sim <- mean(consecutive_sims)

    gaps <- 1 - consecutive_sims
    mean_gap <- mean(gaps)
    max_gap <- max(gaps)

    # Topic drift: consecutive blocks of 3 sentences
    block_size <- 3L
    if (n >= 2 * block_size) {
      n_blocks <- n %/% block_size
      block_centroids <- matrix(0, nrow = n_blocks, ncol = ncol(mat_norm))
      for (b in seq_len(n_blocks)) {
        rows <- ((b - 1) * block_size + 1):min(b * block_size, n)
        block_centroids[b, ] <- colMeans(mat_norm[rows, , drop = FALSE])
      }
      bc_norms <- sqrt(rowSums(block_centroids^2))
      bc_norms[bc_norms == 0] <- 1
      bc_norm <- block_centroids / bc_norms
      block_sims <- rowSums(bc_norm[-n_blocks, , drop = FALSE] * bc_norm[-1, , drop = FALSE])
      topic_drift <- mean(1 - block_sims)
    } else {
      topic_drift <- NA_real_
    }

    # Novelty: cosine distance to running centroid
    cum_mat <- apply(mat_norm, 2, cumsum)
    novelty_scores <- numeric(n - 1)
    for (j in 2:n) {
      running_centroid <- cum_mat[j - 1, ] / (j - 1)
      rc_norm <- sqrt(sum(running_centroid^2))
      if (rc_norm > 0) {
        novelty_scores[j - 1] <- 1 - sum(mat_norm[j, ] * running_centroid) / rc_norm
      } else {
        novelty_scores[j - 1] <- NA_real_
      }
    }
    mean_novelty <- mean(novelty_scores, na.rm = TRUE)

    # n_topics: optimal k via silhouette (min 3 sentences per cluster)
    k_max <- min(5L, n %/% 3L)
    if (k_max >= 2) {
      d_mat <- dist(mat_norm)
      best_k <- 1L
      best_sil <- 0  # silhouette of k=1 is 0 by convention
      for (k in 2:k_max) {
        km <- tryCatch(kmeans(mat_norm, centers = k, nstart = 3, iter.max = 20),
                       warning = function(w) kmeans(mat_norm, centers = k, nstart = 3, iter.max = 20))
        sil <- mean(cluster::silhouette(km$cluster, d_mat)[, "sil_width"])
        if (sil > best_sil) { best_sil <- sil; best_k <- k }
      }
      n_topics <- best_k
    } else {
      n_topics <- 1L
    }

    # Convexity (Gärdenfors): sample sentence-pair midpoints and ask whether
    # they remain supported by the document's other sentence embeddings.
    if (n >= 3) {
      support_radius <- .sentence_support_radius(mat_norm)
      pair_idx <- .sample_sentence_pairs(n, max_pairs = 500L)
      midpoint_support <- .midpoint_support_scores(mat_norm, pair_idx[, 1], pair_idx[, 2])
      midpoint_dist <- 1 - midpoint_support
      knn_radii <- .sentence_knn_radii(mat_norm, k_neighbors = 10L)
      segment_scores <- .segment_support_scores(
        mat_norm, pair_idx[, 1], pair_idx[, 2], knn_radii$radii
      )

      convexity <- mean(midpoint_support, na.rm = TRUE)
      blob_convexity <- mean(midpoint_dist < support_radius, na.rm = TRUE)
      segment_support <- mean(segment_scores$support, na.rm = TRUE)
      segment_occupancy <- mean(segment_scores$occupancy, na.rm = TRUE)
    } else {
      convexity <- NA_real_
      blob_convexity <- NA_real_
      segment_support <- NA_real_
      segment_occupancy <- NA_real_
      support_radius <- NA_real_
      knn_radii <- list(radii = rep(NA_real_, n), effective_k = 0L)
    }

    # Local convexity: same midpoint support test, but consecutive pairs only.
    if (n >= 3) {
      local_midpoint_support <- .midpoint_support_scores(mat_norm, seq_len(n - 1), seq(2L, n))
      local_midpoint_dist <- 1 - local_midpoint_support
      local_segment_scores <- .segment_support_scores(
        mat_norm, seq_len(n - 1), seq(2L, n), knn_radii$radii
      )

      local_convexity <- mean(local_midpoint_support, na.rm = TRUE)
      local_blob_convexity <- mean(local_midpoint_dist < support_radius, na.rm = TRUE)
      local_segment_support <- mean(local_segment_scores$support, na.rm = TRUE)
      local_segment_occupancy <- mean(local_segment_scores$occupancy, na.rm = TRUE)
    } else {
      local_convexity <- NA_real_
      local_blob_convexity <- NA_real_
      local_segment_support <- NA_real_
      local_segment_occupancy <- NA_real_
    }

    results[[i]] <- data.table(
      doc_id = doc,
      emb_thematic_dispersion = mean_centroid_distance,
      emb_centroid_distance_sd = centroid_dist_sd,
      emb_sequential_similarity = mean_sequential_sim,
      emb_mean_semantic_gap = mean_gap,
      emb_max_semantic_gap = max_gap,
      emb_topic_drift = topic_drift,
      emb_mean_novelty = mean_novelty,
      emb_n_topics = n_topics,
      emb_convexity = convexity,
      emb_blob_convexity = blob_convexity,
      emb_segment_support = segment_support,
      emb_segment_occupancy = segment_occupancy,
      emb_local_convexity = local_convexity,
      emb_local_blob_convexity = local_blob_convexity,
      emb_local_segment_support = local_segment_support,
      emb_local_segment_occupancy = local_segment_occupancy
    )
  }

  rbindlist(results)
}
