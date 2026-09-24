# alsi-ai

Python/LLM-backed feature extensions for [ALSI](https://github.com/gloignon/alsi) (Analyseur Lexico-Syntaxique Intégré / Integrated Lexico-Syntactic Analyzer).

ALSI itself is pure R + UDPipe — no Python required. This repo holds everything that needs a Python backend (via [reticulate](https://rstudio.github.io/reticulate/)): transformer-based surprisal and embeddings, spaCy/Trankit parsing, dependency-tree-edit-distance syntax similarity, and Ollama LLM querying. It operates on the same `data.table` corpus format ALSI produces (`build_corpus()` / `parse_text()` / `post_process_lexicon()`), so most workflows start in ALSI and hand off the parsed corpus here.

## Setup

Clone this repo alongside `alsi` (as sibling directories):

```
some_folder/
├── alsi/
└── alsi-ai/
```

Demo scripts `source()` ALSI's classic feature files from `../alsi/R/` for the parsing step, then load this repo's `R/` files for the AI features. Python dependencies (`torch`, `transformers`, `sentence-transformers`, `spacy`, `trankit`, ...) are declared per-function via `reticulate::py_require()` and installed automatically on first use — no manual venv setup needed.

Several demos (inherited as-is from ALSI, where they were originally written) still resolve `models/`, `demo_corpora/`, and cached `out/*.Rds` files relative to the working directory — e.g. `models/french_gsd-remix_3.udpipe` for parsing, or `out/demo_parsed_tagged.Rds` produced by ALSI's `demos/demo_parse_tag.R`. The simplest fix, until these are fully decoupled, is to symlink ALSI's shared resources into this repo before running demos from here:

```sh
mkdir -p models out
ln -s ../../alsi/models/french_gsd-remix_3.udpipe models/french_gsd-remix_3.udpipe
ln -s ../../alsi/models/pos_trigram_fr_gsd_alsi.Rds models/pos_trigram_fr_gsd_alsi.Rds
ln -s ../alsi/demo_corpora demo_corpora
ln -s ../alsi/out/demo_parsed_tagged.Rds out/demo_parsed_tagged.Rds  # after running alsi's demo_parse_tag.R
```

(alsi-ai's own AI models — spaCy, Trankit, the neural POS LM — are fetched separately into this repo's own `models/` directory by `R/artefact_builders/fetch_spacy_models.R` / `fetch_trankit_models.R`, so only the specific ALSI files above are symlinked, not the whole directory.)

`main_ai.R` does not need this — it resolves ALSI's paths explicitly via `ALSI_DIR`.

## Feature families

- **spaCy / Trankit parsing** — `parse_text_spacy()`, `parse_text_trankit()` in `R/fnt_corpus_backends.R`: alternatives to ALSI's default UDPipe backend.
- **syntok sentence segmentation** — `segment_sentences_syntok()`, also in `R/fnt_corpus_backends.R`.
- **Semantic embeddings and coherence** — sentence/document embeddings and thematic dispersion, sequential similarity, topic drift, novelty, and conceptual convexity (`R/fnt_embeddings.R`).
- **LLM surprisal** — token-level surprisal and entropy from masked (MLM) or autoregressive (AR) language models (`R/fnt_surprisal.R`).
- **Neural POS surprisal** — a GRU language model over UPOS sequences, conditioned on full left context (`R/fnt_pos_surprisal_nn.R`); the trigram counterpart lives in ALSI.
- **Top-k word predictions** — beam-search word completion for masked/autoregressive LM predictions (`R/fnt_top_predictions.R`).
- **Dependency-tree-edit-distance syntax similarity** — tree-edit distance via `apted` (`R/fnt_syntax_similarity.R`).
- **Ollama LLM querying** — general-purpose row-by-row querying of a locally-run LLM for annotation, classification, paraphrase, or any templated task (`R/fnt_ollama.R`).

See [FEATURES.md](FEATURES.md) for the full column-by-column inventory, and ALSI's own [FEATURES.md](https://github.com/gloignon/alsi/blob/main/FEATURES.md) for the classic feature families.

## Demos

`demos/` mirrors ALSI's demo scripts:

- `demo_backends.R` — spaCy/Trankit parsing.
- `demo_embeddings.R` — sentence/document embeddings and coherence.
- `demo_surprisal.R`, `demo_cefle_surprisal.R` — LLM surprisal/entropy.
- `demo_pos_surprisal.R` — trigram vs. neural POS surprisal comparison (sources both repos).
- `demo_ollama.R` — Ollama querying.

`main_ai.R` runs the full AI pipeline on ALSI's bundled demo corpus, picking up where ALSI's own `main.R` leaves off.

## One-off tooling

`scripts/` holds data-acquisition scripts (e.g. `scrape_alector.py`) that were used once to build resources ALSI now ships pre-built. They are not part of the runtime feature pipeline.

## License

Same terms as ALSI — see [LICENSE.md](LICENSE.md). Third-party Python packages and models (transformers, spaCy, Trankit, Ollama models, etc.) are governed by their own licenses.
