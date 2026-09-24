# embed_sentences.py
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import torch

# Global model variable
model = None

# Pick best available device across platforms
def _best_device():
    if os.environ.get("ALSIO_FORCE_CPU") in ("1", "true", "TRUE", "yes", "YES"):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Load a model
def load_embedding_model(model_name="Lajavaness/bilingual-embedding-large"):
    global model
    device = _best_device()
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True
    )
    print(f"Loaded model: {model_name}")
    print(f"Device: {model.device}")
        # print the parameters that this model can take when doing model.encode:
    print(f"Model parameters: {model.encode.__code__.co_varnames}")
    print("CUDA available:", torch.cuda.is_available())
    if hasattr(torch.backends, "mps"):
        print("MPS available:", torch.backends.mps.is_available())

# Embed the sentences with optional instruction prefix
def embed_sentences(sentences, normalize=False, mode = "basic", instruction=None, bs=32):
    """
    Given a list of sentences, return a NumPy matrix of sentence embeddings.
    If `instruction` is provided, it will be prepended to each sentence.
    
    Assumes model is already loaded via load_embedding_model().
    """
    if model is None:
        raise RuntimeError("Model not loaded. Call load_embedding_model() first.")
    
    bs = int(bs)

    # if instruction is empty, set to None
    instruction = instruction if instruction else None
    
    # check if model has a prompt parameter
    if mode == "prompt" and not hasattr(model, "encode"):
      raise ValueError("Model does not support the prompt= parameter. Use mode='query' instead.")
  
    # if mode is "basic" or instruction set to None...
    if mode == "basic" or instruction is None:
      return model.encode(
      sentences,
      show_progress_bar=False,
      convert_to_numpy=True,
      batch_size = bs,
      normalize_embeddings=normalize
    )
    #else if mode is "prompt" we will use the prompt parameter
    # e.g. for KaLM
    elif mode == "prompt":
      return model.encode(
      sentences,
      prompt = instruction,
      show_progress_bar=False,
      convert_to_numpy=True,
      batch_size=bs,
      normalize_embeddings=normalize
    )
    # else if mode is query we will construct the query using instructio + sentence
    elif mode == "query":
      text_input = [instruction + sentence for sentence in sentences]
      return model.encode(text_input,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=bs,
        normalize_embeddings=normalize
      )
    # other modes will return an error
    else:
        raise ValueError("Invalid mode. Choose from 'basic', 'prompt', or 'query'.")
