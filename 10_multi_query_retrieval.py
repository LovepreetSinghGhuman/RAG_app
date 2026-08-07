import sys
import warnings
from pathlib import Path
import json
import re

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


# all_retrieval_results = [
#     [Doc1, Doc2, Doc3, Doc4, Doc5],  ← Query 1 results
#     [Doc2, Doc1, Doc6, Doc7, Doc3],  ← Query 2 results  
#     [Doc8, Doc2, Doc9, Doc10, Doc11] ← Query 3 results
# ]