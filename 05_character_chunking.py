import os
import warnings
import importlib
import torch

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import CharacterTextSplitter
 
warnings.filterwarnings("ignore")
load_dotenv()

module = importlib.import_module("01_ingestion_pipline")
load_docs = module.load_docs

# --- Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else fallback to CPU
DB_PATH = os.getenv("VECTOR_DB_PATH", "db/chroma")  # Vector DB storage path, overridable via .env

# 5 types of chunking/splitting strategies: "Recursive", "Character", "Document-Specific", "Semantic", "Agentic"
# Here we are implementing the "Character" chunking strategy, which splits documents into smaller pieces based on character count.
# This is useful for processing large documents that may exceed model input limits or for creating more manageable chunks for embedding and retrieval.

def chunk_documents_character(docs_path, chunk_size=1000, chunk_overlap=0):
    print("Chunking documents into smaller pieces ...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(docs_path)

    print(f"Total chunks created: {len(chunks)}")
    return chunks

def main():
    if DEVICE == "cuda":
        print("Using GPU for embeddings.")
    else:
        print("Using CPU for embeddings.")

    # 1. Load files
    documents = load_docs(docs_path="docs")

    # 2. Chunking the files into smaller pieces and printing
    if documents:
        chunks = chunk_documents_character(documents)
        if chunks:
            print("Example chunk:")
            print(chunks[0].page_content[:1000])
            print("-" * 50)
            print(chunks[1].page_content[:1000])
            print("-" * 50)
    else:
        print("No documents to process. Exiting.")
        return


if __name__ == "__main__":
    main()