# alsi-ai Feature Inventory

Scope: Python/LLM-backed feature families. For classic (R/UDPipe-only)
features, see ALSI's own
[FEATURES.md](https://github.com/gloignon/alsi/blob/main/FEATURES.md).

Scripts covered:
- `R/fnt_pos_surprisal_nn.R`
- `R/fnt_embeddings.R`
- `R/fnt_surprisal.R`
- `R/fnt_top_predictions.R`
- `R/fnt_syntax_similarity.R`
- `R/fnt_ollama.R`
- `R/fnt_corpus_backends.R`
- `main_ai.R`

## 1) `features$pos_surprisal_nn`
Produced in: `R/fnt_pos_surprisal_nn.R` (`pos_surprisal_nn`)

A GRU language model over UPOS sequences, conditioned on the full left
context (unlike ALSI's fixed-order trigram `pos_surprisal()`). Columns are
suffixed `_nn` so results can coexist with the trigram measures.

| Feature name | Short description | Level |
|---|---|---|
| `pos_surprisal_nn` | Token-level neural POS surprisal (−log₂ p). | word |
| `pos_entropy_nn` | Token-level predictive entropy from the neural POS LM. | word |

## 2) `features$embeddings$dt_sent_embeddings`
Produced in: `R/fnt_embeddings.R` (`encode_embeddings`, `corpus_embeddings`)

| Feature name | Short description | Level |
|---|---|---|
| `dim1` ... `dimN` | Sentence embedding dimensions (N depends on embedding model). | sentence |

## 3) `features$embeddings$dt_doc_embeddings`
Produced in: `R/fnt_embeddings.R` (`corpus_embeddings`)

| Feature name | Short description | Level |
|---|---|---|
| `dim1` ... `dimN` | Document-level mean of sentence embedding dimensions. | document |

## 4) `features$embedding_coherence`
Produced in: `R/fnt_embeddings.R` (`embedding_coherence`)

Input: sentence embeddings from `corpus_embeddings()`. Returns one row per document.

| Feature name | Short description | Level |
|---|---|---|
| `emb_thematic_dispersion` | Mean cosine distance from each sentence to the document centroid. | document |
| `emb_centroid_distance_sd` | SD of cosine distances to centroid. | document |
| `emb_sequential_similarity` | Mean cosine similarity between consecutive sentences. | document |
| `emb_mean_semantic_gap` | Mean cosine distance between consecutive sentences. | document |
| `emb_max_semantic_gap` | Largest cosine distance between consecutive sentences. | document |
| `emb_topic_drift` | Mean cosine distance between consecutive 3-sentence block centroids. | document |
| `emb_mean_novelty` | Mean cosine distance of each sentence to the running centroid of all previous sentences. | document |
| `emb_n_topics` | Optimal number of sentence clusters via silhouette method (k = 1..min(5, n/3)). | document |
| `emb_convexity` | Conceptual convexity (Gärdenfors): mean nearest-neighbour cosine support for sampled sentence-pair midpoints, excluding the pair endpoints. 1 = strongly supported. | document |
| `emb_blob_convexity` | Proportion of sampled sentence-pair midpoints whose nearest non-endpoint sentence is within the document's median nearest-neighbour support radius. | document |
| `emb_segment_support` | Local-scale segment support: mean exp(-0.5 × normalized midpoint distance²), where normalized distance uses the geometric mean of endpoint k-NN radii. | document |
| `emb_segment_occupancy` | Proportion of sampled sentence-pair midpoints within the endpoint k-NN local support scale. | document |
| `emb_local_convexity` | Same midpoint-support score but only on consecutive sentence pairs; measures local semantic continuity. | document |
| `emb_local_blob_convexity` | Local consecutive-pair version of blob convexity. | document |
| `emb_local_segment_support` | Consecutive-pair version of local-scale segment support. | document |
| `emb_local_segment_occupancy` | Consecutive-pair version of local-scale segment occupancy. | document |

## 5) `features$surprisal$mlm` and `features$surprisal$ar`
Produced in: `R/fnt_surprisal.R` (`llm_surprisal_entropy`)

| Feature name | Short description | Level |
|---|---|---|
| `llm_surprisal` | Token-level language-model surprisal (MLM or AR, depending on call). | word |
| `llm_entropy` | Token-level predictive entropy from language model. | word |
| `llm_subword_n` | Number of subword pieces used for the token. | word |

## 6) Top-k word predictions
Provided by: `R/fnt_top_predictions.R`

Beam-search word completion for masked/autoregressive language model
predictions — keeps the top-k predicted words when a prediction is a
subtoken.

## 7) Dependency-tree-edit-distance syntax similarity
Provided by: `R/fnt_syntax_similarity.R`

Tree-edit distance (TED) between dependency trees, computed via the `apted`
Python package.

## 8) `parse_text_spacy()` / `parse_text_trankit()` / `segment_sentences_syntok()`
Provided by: `R/fnt_corpus_backends.R`

Alternative parsing/sentence-segmentation backends to ALSI's default UDPipe
pipeline — same CoNLL-U-style `data.table` output as `alsi::parse_text()`.

## 9) Ollama LLM querying
Provided by: `R/fnt_ollama.R` (`ollama_generate`)

Not a fixed feature set — this is a general-purpose utility for querying a local LLM
(via [Ollama](https://ollama.com)) row-by-row from a data frame. Each row becomes one
stateless API call with a templated prompt; the response is stored in an
`ollama_response` column.

Typical uses: generating annotations, classifications, paraphrases, or verification
labels that can then be parsed into whatever columns are needed downstream.

Requires Ollama running locally (or on the network). See `demos/demo_ollama.R` for a
worked example that generates statements about texts and then verifies them.
