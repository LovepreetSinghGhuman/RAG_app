import os
import warnings
import importlib
import torch

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_experimental.text_splitter import SemanticChunker
 
warnings.filterwarnings("ignore")
load_dotenv()

module = importlib.import_module("01_ingestion_pipline")
load_docs = module.load_docs
create_embeddings_vector = module.create_embeddings_vector

# --- Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else fallback to CPU
DB_PATH = os.getenv("VECTOR_DB_PATH", "db/chroma")  # Vector DB storage path, overridable via .env

# 5 types of chunking/splitting strategies: "Recursive", "Character", "Document-Specific", "Semantic", "Agentic"
# Here we are implementing the "Semantic" chunking strategy, which splits documents into smaller pieces based on semantic similarity.
# This is useful for processing large documents that may exceed model input limits or for creating more manageable chunks for embedding and retrieval.
# Using HuggingFaceEmbeddings instead of OpenAIEmbeddings for better performance and flexibility (offering multiple models and offline use)
embedding_model = HuggingFaceEmbeddings( 
    model_name="BAAI/bge-small-en-v1.5",  # good balance of speed/quality for English text
    model_kwargs={"device": DEVICE},        # change to "cuda" if NVIDIA GPU available; For ROCm check PyTorch ROCm support (rocm.7.2.1 and above)
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},  # recommended for BGE models (cosine similarity)
)

def chunk_documents_semantic(docs_path, chunk_size=1000, chunk_overlap=0):
    print("Chunking documents into smaller pieces ...")

    text_splitter = SemanticChunker(
        embedding_model=embedding_model, # embedding model needed for semantic similarity calculations for "Character" and "Recursive" not needed
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(docs_path)

    print(f"Total chunks created: {len(chunks)}")
    return chunks

def main():
    print(f"Using {'GPU' if DEVICE == 'cuda' else 'CPU'} for embeddings.")

    # 1. Load files
    documents = load_docs(docs_path="docs")

    # 2. Chunking the files into smaller pieces and printing
    chunks = chunk_documents_semantic(documents)
    if not chunks:
            print("No chunks created. Exiting.")
            return
    
    # Preview the first two chunks to sanity-check the splitting output
    print("Example chunks:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"[Chunk {i}]")
        print(chunk.page_content[:1000])
        print("-" * 50)


if __name__ == "__main__":
    main()