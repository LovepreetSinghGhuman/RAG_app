import sys
import warnings
import importlib
from pathlib import Path

# Anchor the import to this script's own location so it works regardless of
# the working directory the script is run from.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from config.config import DEVICE, DB_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_KWARGS

warnings.filterwarnings("ignore")
load_dotenv()

module = importlib.import_module("01_ingestion_pipline")
load_docs = module.load_docs
create_embeddings_vector = module.create_embeddings_vector

# 5 types of chunking/splitting strategies: "Recursive", "Character", "Document-Specific", "Semantic", "Agentic"
# Here we are implementing the "Semantic" chunking strategy, which splits documents into smaller pieces based on semantic similarity.
# This is useful for processing large documents that may exceed model input limits or for creating more manageable chunks for embedding and retrieval.
# Using HuggingFaceEmbeddings instead of OpenAIEmbeddings for better performance and flexibility (offering multiple models and offline use)
#
# model_name and encode_kwargs come from config.config, so they're guaranteed to match
# ingestion_pipline.py exactly. A mismatch here would silently produce a different vector
# space than the one your documents were embedded into, which breaks similarity search.
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},  # change to "cuda" if NVIDIA GPU available; For ROCm check PyTorch ROCm support (rocm.7.2.1 and above)
    encode_kwargs=EMBEDDING_KWARGS,
)


def chunk_documents_semantic(docs):
    print("Chunking documents into smaller pieces ...")

    text_splitter = SemanticChunker(
        embeddings=embedding_model,  # embedding model needed for semantic similarity calculations for "Character" and "Recursive" not needed
        breakpoint_threshold_amount=90,  # threshold for semantic similarity; lower values result in more chunks
        breakpoint_threshold_type="percentile",  # type of similarity metric to use; "cosine" is recommended for BGE models
        min_chunk_size=200,  # minimum chunk size in characters; smaller values result in more chunks
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
    chunks = chunk_documents_semantic(documents)
    if not chunks:
        print("No chunks created. Exiting.")
        sys.exit(0)

    # Preview the first few chunks to sanity-check the splitting output
    print("Example chunks:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"[Chunk {i}]")
        print(chunk.page_content[:1000])
        print("-" * 50)