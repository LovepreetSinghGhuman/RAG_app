import sys
import warnings
import importlib
from pathlib import Path

# Anchor the import to this script's own location so it works regardless of
# the working directory the script is run from.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter

from config.config import DEVICE, DB_PATH

warnings.filterwarnings("ignore")
load_dotenv()

module = importlib.import_module("01_ingestion_pipline")
load_docs = module.load_docs

# 5 types of chunking/splitting strategies: "Recursive", "Character", "Document-Specific", "Semantic", "Agentic"
# Here we are implementing the "Character" chunking strategy, which splits documents into smaller pieces based on character count.
# This is useful for processing large documents that may exceed model input limits or for creating more manageable chunks for embedding and retrieval.

def chunk_documents_character(docs, chunk_size=1000, chunk_overlap=0):
    print("Chunking documents into smaller pieces ...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    print(f"Using {'GPU' if DEVICE == 'cuda' else 'CPU'} for embeddings.")

    # 1. Load files
    documents = load_docs(docs_path="docs")
    if not documents:
        print("No documents to process. Exiting.")
        sys.exit(0)

    # 2. Chunking the files into smaller pieces and printing
    chunks = chunk_documents_character(documents)
    if not chunks:
        print("No chunks created. Exiting.")
        sys.exit(0)

    # Preview the first few chunks to sanity-check the splitting output
    print("Example chunks:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"[Chunk {i}]")
        print(chunk.page_content[:1000])
        print("-" * 50)