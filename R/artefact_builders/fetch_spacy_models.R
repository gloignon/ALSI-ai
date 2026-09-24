# Fetch the spaCy French model, hosted as a GitHub release asset.
#
# The model is too large to keep in regular git history, so it is
# distributed as a zip on the "models-v1" GitHub release instead. This
# script downloads and unpacks it into models/ if missing.
#
# Run from the repo root:
#   Rscript R/artefact_builders/fetch_spacy_models.R

RELEASE_BASE <- "https://github.com/gloignon/ALSI/releases/download/models-v1"

models <- list(
  "spacy_fr_gsd_alsi_v1" = file.path(RELEASE_BASE, "spacy_fr_gsd_alsi_v1.zip")
)

dir.create("models", showWarnings = FALSE, recursive = TRUE)

for (name in names(models)) {
  local_path <- file.path("models", name)
  if (dir.exists(local_path)) {
    message("Already present: ", local_path)
    next
  }
  message("Downloading ", name, "...")
  zip_path <- tempfile(fileext = ".zip")
  download.file(models[[name]], zip_path, mode = "wb", quiet = TRUE)
  unzip(zip_path, exdir = "models")
  unlink(zip_path)
}
