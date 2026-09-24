# Fetch the Trankit French model, hosted as a GitHub release asset.
#
# Only the fine-tuned "customized-mwt" model needs to be fetched (~34 MB).
# The XLM-RoBERTa base model (~2 GB) is downloaded automatically from
# HuggingFace by trankit on first use and cached under the same directory.
#
# Run from the repo root:
#   Rscript R/artefact_builders/fetch_trankit_models.R

RELEASE_BASE <- "https://github.com/gloignon/ALSI/releases/download/models-v1"

models <- list(
  "trankit_fr_v1/xlm-roberta-base/customized-mwt" =
    file.path(RELEASE_BASE, "trankit_fr_v1_customized-mwt.zip")
)

for (name in names(models)) {
  local_path <- file.path("models", name)
  if (dir.exists(local_path)) {
    message("Already present: ", local_path)
    next
  }
  message("Downloading ", basename(name), "...")
  exdir <- dirname(local_path)
  dir.create(exdir, showWarnings = FALSE, recursive = TRUE)
  zip_path <- tempfile(fileext = ".zip")
  download.file(models[[name]], zip_path, mode = "wb", quiet = TRUE)
  unzip(zip_path, exdir = exdir)
  unlink(zip_path)
}
