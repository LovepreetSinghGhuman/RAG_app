import os
import torch
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Using AOTriton backend for Efficient Attention")
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# SETUP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise fallback to CPU
DBPATH = os.getenv("VECTOR_DB_PATH", "db/chroma")  # Default path env or "db/chroma"

emmbedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # same model as in ingestion_pipeline.py for consistency (1 model for RAG to avoid embedding mismatch)
    model_kwargs={"device": DEVICE}
)

db = Chroma(
    persist_directory=DBPATH, 
    embedding_function=emmbedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search query for testing the retrieval pipeline
query = "In what year did Tesla begin production of the Roadster?"  # Example query (testing)

# retriever = db.as_retriever(search_kwargs={"k": 3}) <-- alternative for code below (good for testing, but not as effective as similarity search for RAG)

retriever = db.as_retriever(
    # Use similarity search for better results
    search_type="similarity_score_threshold",
    # Retrieve top 5 documents with a similarity score above 0.3
    search_kwargs={
        "k": 5, 
        "score_threshold": 0.3
    }  
)

relevant_docs = retriever.invoke(query)

print(f"Query: {query}")
if relevant_docs:
    print(f"Found {len(relevant_docs)} relevant documents:")
    for i, doc in enumerate(relevant_docs):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown source')}")
        print(f"Content: {doc.page_content[:500]}...")  # Print first 500 characters of the content