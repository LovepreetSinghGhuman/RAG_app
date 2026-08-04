import os
import torch
import warnings

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
 
warnings.filterwarnings("ignore")
load_dotenv()

# --- Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else fallback to CPU
DB_PATH = os.getenv("VECTOR_DB_PATH", "db/chroma")  # Vector DB storage path, overridable via .env
 
# IMPORTANT: model_name and encode_kwargs must match ingestion_pipline.py exactly.
# A mismatch here silently produces a different vector space than the one your documents were embedded into, which breaks similarity search.
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # must match ingestion_pipline.py
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},  # must match ingestion_pipline.py
)
 
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)


#  Search query for testing the retrieval pipeline
# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
