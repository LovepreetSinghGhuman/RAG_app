import sys
import warnings
from pathlib import Path
import json
import re
from collections import defaultdict

# Anchor the import to this script's own location so it works regardless of
# the working directory the script is run from.
sys.path.insert(0, str(Path(__file__).parent))

import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from pydantic import BaseModel
from typing import List

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

chat = ChatHuggingFace(llm=llm)

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]
# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents.

Original query: {original_query}

Return ONLY a JSON object in this exact format, with no other text:
{{"queries": ["variation 1", "variation 2", "variation 3"]}}"""

response = chat.invoke(prompt)
raw_output = response.content.strip()

# Models sometimes wrap JSON in ```json fences or add stray text — strip that defensively
match = re.search(r"\{.*\}", raw_output, re.DOTALL)
if not match:
    raise ValueError(f"Could not find JSON in model output:\n{raw_output}")

parsed = json.loads(match.group())
query_variations_obj = QueryVariations(**parsed)  # validates shape via Pydantic
query_variations = query_variations_obj.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION (RRF) is a method to combine results from multiple queries. It gives higher scores to documents that appear in multiple query results, while also considering their rank positions.
# Formula: RRF score = sum(1 / (k + rank_position)) for each document across all queries, where k is a constant (e.g., 60) to control the influence of rank position.

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")


print("Notice how different query variations retrieved different documents.")

# ──────────────────────────────────────────────────────────────────
# Step 3: Apply Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks

# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)

# ──────────────────────────────────────────────────────────────────
# Step 4: Display Final Fused Results
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations.")
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score") 
print("   • Balanced fusion using k=60 for gentle position penalties")

# ──────────────────────────────────────────────────────────────────
# Optional: Quick Usage Examples
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("USAGE EXAMPLES")
print("="*60)