# config.py
import os
import torch

# --- Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = os.getenv("VECTOR_DB_PATH", "db/chroma")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_KWARGS = {"normalize_embeddings": True, "batch_size": 32}