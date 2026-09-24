# fnt_ollama.R
# Query a local Ollama (or Ollama-compatible) API row-by-row from a data frame.
# Uses a chat endpoint with system prompt per request (stateless, no conversation memory).
#
# Two API flavours are supported via the `api` argument of ollama_generate():
#   - "ollama" (default): native Ollama /api/chat, response in $message$content.
#     Connectivity is checked via /api/version.
#   - "openai": OpenAI-compatible /v1/chat/completions (e.g. llama.cpp server,
#     vLLM, LM Studio), response in $choices[[1]]$message$content.
#     Connectivity is checked via /v1/models.

#' Fill a prompt template with column values from one row (internal)
#'
#' Replaces \code{{column_name}} tags in \code{template} with the
#' corresponding values from a single data.frame row.
#'
#' @param template Character string with \code{{column_name}} placeholders.
#' @param row A single-row data.frame or list.
#' @returns A character string with tags replaced.
#' @keywords internal
.fill_template <- function(template, row) {
  out <- template
  for (col in names(row)) {
    tag <- paste0("{", col, "}")
    if (grepl(tag, out, fixed = TRUE)) {
      out <- gsub(tag, as.character(row[[col]]), out, fixed = TRUE)
    }
  }
  out
}

#' Send a single stateless chat request to an LLM server (internal)
#'
#' Posts a system + user message pair to either a native Ollama
#' \code{/api/chat} endpoint or an OpenAI-compatible \code{/v1/chat/completions}
#' endpoint, and returns the assistant response content.
#'
#' @param model Model name.
#' @param system_prompt System prompt string.
#' @param user_prompt User prompt string.
#' @param options Named list of model options (temperature, top_p, num_ctx).
#' @param endpoint API base URL.
#' @param api Either "ollama" (native API) or "openai" (OpenAI-compatible API).
#' @returns A character string: the model response.
#' @keywords internal
.ollama_chat <- function(model, system_prompt, user_prompt, options, endpoint, api = "ollama") {
  messages <- list(
    list(role = "system",  content = system_prompt),
    list(role = "user",    content = user_prompt)
  )

  if (api == "openai") {
    body <- list(
      model       = model,
      messages    = messages,
      stream      = FALSE,
      temperature = options$temperature,
      top_p       = options$top_p
    )
    url <- paste0(endpoint, "/v1/chat/completions")
  } else {
    body <- list(
      model    = model,
      messages = messages,
      stream   = FALSE,
      options  = options
    )
    url <- paste0(endpoint, "/api/chat")
  }

  res <- httr::POST(
    url,
    body = jsonlite::toJSON(body, auto_unbox = TRUE),
    encode = "raw",
    httr::content_type_json(),
    httr::timeout(600)
  )
  if (httr::status_code(res) != 200) {
    stop(sprintf("HTTP %d: %s", httr::status_code(res),
                 httr::content(res, as = "text", encoding = "UTF-8")))
  }
  parsed <- jsonlite::fromJSON(httr::content(res, as = "text", encoding = "UTF-8"),
                               simplifyVector = FALSE)

  if (api == "openai") {
    return(parsed$choices[[1]]$message$content)
  }
  return(parsed$message$content)
}


#' Query Ollama row-by-row from a data frame.
#'
#' @param data               Data frame whose columns fill the prompt template.
#' @param user_prompt_template  String with \code{{column_name}} placeholders.
#' @param system_prompt       System prompt sent with every request.
#' @param model               Ollama model name (e.g. "gemma3:4b").
#' @param temperature         Sampling temperature (default 0.7).
#' @param top_p               Nucleus-sampling cutoff (default 0.9).
#' @param num_ctx             Context window in tokens (default 4096).
#' @param output_file         Path for incremental CSV output (also used for resuming).
#' @param force_restart       If TRUE, ignore existing output file and start from row 1.
#' @param endpoint            API base URL (Ollama server, or any Ollama-compatible server).
#' @param api                 Either "ollama" (native Ollama API, default) or "openai"
#'                            (OpenAI-compatible API, e.g. llama.cpp server, vLLM, LM Studio).
#'                            Use "openai" when \code{endpoint} is not a real Ollama server.
#' @returns A data.table: all original columns plus \code{ollama_response}.
ollama_generate <- function(data,
                            user_prompt_template,
                            system_prompt,
                            model = "llama3",
                            temperature = 0.7,
                            top_p = 0.9,
                            num_ctx = 4096,
                            output_file = NULL,
                            force_restart = FALSE,
                            endpoint = "http://localhost:11434",
                            api = c("ollama", "openai")) {

  api <- match.arg(api)

  dt <- as.data.table(data)
  n <- nrow(dt)

  # Validate template tags
  tags_found <- regmatches(user_prompt_template,
                           gregexpr("\\{([^}]+)\\}", user_prompt_template))[[1]]
  tag_names <- gsub("[{}]", "", tags_found)
  missing_cols <- setdiff(tag_names, names(dt))
  if (length(missing_cols)) {
    stop(sprintf("[ollama] Template tags not found in data columns: %s",
                 paste(missing_cols, collapse = ", ")))
  }

  # Resume from previous run if output_file exists
  start_row <- 1L
  if (!is.null(output_file) && file.exists(output_file) && !force_restart) {
    dt_prev <- fread(output_file, encoding = "UTF-8")
    completed <- which(!is.na(dt_prev$ollama_response))
    if (length(completed) > 0) {
      start_row <- max(completed) + 1L
      dt[, ollama_response := NA_character_]
      dt$ollama_response[seq_len(nrow(dt_prev))] <- dt_prev$ollama_response
      message(sprintf("[ollama] Resuming from row %d/%d.", start_row, n))
    }
  }

  if (!"ollama_response" %in% names(dt)) {
    dt[, ollama_response := NA_character_]
  }

  if (start_row > n) {
    message("[ollama] All rows already completed.")
    return(dt)
  }

  options <- list(temperature = temperature, top_p = top_p, num_ctx = num_ctx)

  # Check connectivity (probe path differs by API flavour)
  ping_path <- if (api == "openai") "/v1/models" else "/api/version"
  ping <- tryCatch(httr::GET(paste0(endpoint, ping_path), httr::timeout(5)),
                   error = function(e) NULL)
  if (is.null(ping) || httr::status_code(ping) != 200) {
    stop(sprintf("[ollama] Cannot reach %s server at %s. Is it running?", api, endpoint))
  }

  message(sprintf("[ollama] Querying model '%s' on %d row(s)...", model, n - start_row + 1L))

  start_time <- Sys.time()
  n_todo <- n - start_row + 1L
  n_done <- 0L
  errors <- list()

  for (i in start_row:n) {
    prompt_i <- .fill_template(user_prompt_template, dt[i])

    result <- tryCatch(
      .ollama_chat(model, system_prompt, prompt_i, options, endpoint, api),
      error = function(e) e
    )

    if (inherits(result, "error")) {
      msg <- conditionMessage(result)
      errors[[length(errors) + 1]] <- list(row = i, message = msg)
      warning(sprintf("[ollama] Row %d FAILED: %s", i, msg), immediate. = TRUE)
    } else if (is.null(result) || !nzchar(result)) {
      errors[[length(errors) + 1]] <- list(row = i, message = "empty response")
      warning(sprintf("[ollama] Row %d returned empty response", i), immediate. = TRUE)
    } else {
      set(dt, i, "ollama_response", result)
    }

    n_done <- n_done + 1L
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    if (elapsed > 0) {
      speed <- n_done / elapsed
      remaining <- (n_todo - n_done) / speed
      cat(sprintf("\r[ollama] [%d/%d] %.1f rows/min — ~%.1f min left     ",
                  i, n, speed * 60, remaining / 60))
    } else {
      cat(sprintf("\r[ollama] [%d/%d] ...     ", i, n))
    }
    flush.console()

    if (!is.null(output_file)) fwrite(dt, output_file)
  }

  cat("\n")

  if (length(errors)) {
    message(sprintf("[ollama] %d row(s) failed:", length(errors)))
    for (e in errors) message(sprintf("  row %d: %s", e$row, e$message))
  }

  elapsed_total <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  message(sprintf("[ollama] Done. %d/%d rows completed in %.1f min.", n - length(errors), n, elapsed_total))

  return(dt)
}

#' Parse a numbered list response from an LLM into a character vector
#'
#' Extracts items from a response formatted as a numbered list
#' (e.g. "1. First item\n2. Second item"). Missing items become \code{NA}.
#'
#' @param response A single character string (one LLM response).
#' @param n Number of items expected.
#' @returns A character vector of length \code{n}.
parse_numbered_list <- function(response, n = 4L) {
  if (is.na(response)) return(rep(NA_character_, n))
  lines <- unlist(strsplit(response, "\n"))
  purrr::map_chr(seq_len(n), function(k) {
    pattern <- sprintf("^\\s*\\*{0,2}%d[.):]+\\*{0,2} *", k)
    hit <- grep(pattern, lines, value = TRUE)
    if (length(hit) > 0) trimws(sub(pattern, "", hit[1])) else NA_character_
  })
}
