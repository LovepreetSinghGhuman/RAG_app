import os
import torch

# --- Shared settings for ingestion + retrieval pipelines ---
# IMPORTANT: DEVICE doesn't need to match between runs, but EMBEDDING_MODEL_NAME
# and EMBEDDING_KWARGS DO need to match exactly between ingestion and retrieval.
# A mismatch here silently produces a different vector space than the one your
# documents were embedded into, which breaks similarity search.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else fallback to CPU
DB_PATH = os.getenv("VECTOR_DB_PATH", "db/chroma")  # Vector DB storage path, overridable via .env

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # good balance of speed/quality for English text
EMBEDDING_KWARGS = {
    "normalize_embeddings": True,  # recommended for BGE models (cosine similarity)
    "batch_size": 32,
}