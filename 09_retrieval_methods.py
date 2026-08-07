import sys
import warnings
from pathlib import Path

# Anchor the import to this script's own location so it works regardless of
# the working directory the script is run from.
sys.path.insert(0, str(Path(__file__).parent))

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline

from config.config import DEVICE, DB_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_KWARGS

warnings.filterwarnings("ignore")
load_dotenv()

# --- Setup ---
# model_name and encode_kwargs come from config.config, so they're guaranteed
# to match ingestion_pipline.py exactly. A mismatch here would silently
# produce a different vector space than the one your documents were embedded
# into, which breaks similarity search.
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs=EMBEDDING_KWARGS,
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)


def build_chat_model():
    # Only needed if you extend this script to generate an answer from the
    # retrieved docs. None of the retrieval methods below use it, so it's
    # built lazily rather than at import time — loading the 7B model just to
    # test a retriever is wasted VRAM/time.
    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        device_map="auto",
        model_kwargs={"dtype": torch.float16},          # load weights in fp16 to fit VRAM
        pipeline_kwargs={
            "temperature": 0.2,
            "do_sample": True,          # required for temperature to have any effect
            "max_new_tokens": 512,      # explicit cap, avoids conflicting with the model's default max_length=20
            "return_full_text": False,  # return ONLY the generated answer, not prompt+answer glued together
        },
    )
    return ChatHuggingFace(llm=llm)


def print_docs(docs):
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:")
        print(f"{doc.page_content}\n")


if __name__ == "__main__":
    print(f"Using {'GPU' if DEVICE == 'cuda' else 'CPU'} for compute.")

    # Query to test
    query = "How much did Microsoft pay to acquire GitHub?"
    # query = "How do you plant tomatoes in a garden?"
    print(f"Query: {query}\n")

    # ──────────────────────────────────────────────────────────────────
    # METHOD 1: Basic Similarity Search
    # Returns the top k most similar documents
    # ──────────────────────────────────────────────────────────────────

    # print("=== METHOD 1: Similarity Search (k=3) ===")
    # retriever = db.as_retriever(search_kwargs={"k": 3})
    # docs = retriever.invoke(query)
    # print(f"Retrieved {len(docs)} documents:\n")
    # print_docs(docs)
    # print("-" * 60)

    # ──────────────────────────────────────────────────────────────────
    # METHOD 2: Similarity with Score Threshold
    # Only returns documents above a certain similarity score
    # ──────────────────────────────────────────────────────────────────

    # print("\n=== METHOD 2: Similarity with Score Threshold ===")
    # retriever = db.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={
    #         "k": 3,
    #         "score_threshold": 0.3  # Only return docs with similarity >= 0.3
    #     }
    # )
    # docs = retriever.invoke(query)
    # print(f"Retrieved {len(docs)} documents (threshold: 0.3):\n")
    # print_docs(docs)
    # print("-" * 60)

    # ──────────────────────────────────────────────────────────────────
    # METHOD 3: Maximum Marginal Relevance (MMR)
    # Balances relevance and diversity - avoids redundant results
    # ──────────────────────────────────────────────────────────────────

    print("\n=== METHOD 3: Maximum Marginal Relevance (MMR) ===")
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,             # Final number of docs
            "fetch_k": 10,      # Initial pool to select from
            "lambda_mult": 0.5  # 0=max diversity, 1=max relevance
        }
    )

    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents (\u03bb=0.5):\n")
    print_docs(docs)

    print("=" * 60)
    print("Done! Try different queries or parameters to see the differences.")