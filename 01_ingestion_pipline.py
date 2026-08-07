import os
import torch
import warnings

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# --- Setup ---
from config.config import DEVICE, DB_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_KWARGS


def load_docs(docs_path="docs"):
    print("Loading documents from the directory...")

    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        print(f"Data directory not found: {docs_path}")
        return
    
    # Load all text files from the specified directory
    loader = DirectoryLoader(
        docs_path, 
        glob="**/*.txt", 
        loader_cls=TextLoader, # TextLoader is used to load .txt files. You can change this to another loader if you have different file types. (WebLoader, PDFLoader, etc.)
        loader_kwargs={"encoding": "utf-8"},  # Specify encoding (windows error default is 'cp1252', need 'utf-8' for most text files)
    )

    docs = loader.load()

    if len(docs) == 0:
        print(f"No text files found in the directory: {docs_path}")
        return
    
    # for i, doc in enumerate(docs):
    #     print(f"Document {i+1}: {doc.metadata.get('source', 'Unknown source')} - {len(doc.page_content)} characters")

    print("Documents loaded successfully.")
    return docs


def chunk_documents(docs_path, chunk_size=800, chunk_overlap=0):
    print("Chunking documents into smaller pieces ...")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],  # Splitting on paragraphs > lines > spaces > characters
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(docs_path) # Using .split_documents() instead of .split_text() (better for handling metadata/traceability and document structure)

    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_embeddings_vector(chunks, vector_db_path="db/chroma"):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs=EMBEDDING_KWARGS,
    )

    if os.path.exists(vector_db_path) and os.listdir(vector_db_path):
        print(f"Vector DB already exists at {vector_db_path} — skipping ingestion.")
        return Chroma(
            persist_directory=vector_db_path,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

    print("Creating embeddings for the chunks...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=vector_db_path,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vector_db


print(f"Using {'GPU' if DEVICE == 'cuda' else 'CPU'} for embeddings.")

# 1. Load raw documents from the docs folder
documents = load_docs(docs_path="docs")
if not documents:
    print("No documents to process. Exiting.")
    pass

# 2. Split documents into smaller, fixed-size chunks
chunks = chunk_documents(documents)
if not chunks:
    print("No chunks created. Exiting.")
    pass

# Preview the first two chunks to sanity-check the splitting output
print("Example chunks:")
for i, chunk in enumerate(chunks[:5]):
    print(f"[Chunk {i}]")
    print(chunk.page_content[:1000])
    print("-" * 50)

# 3. Generate embeddings and store chunks in the vector database
print("Creating embeddings and storing them in the vector database...")
vector_db = create_embeddings_vector(chunks, vector_db_path=DB_PATH)
