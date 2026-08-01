import os
import torch
import warnings

warnings.filterwarnings("ignore", message="Using AOTriton backend for Efficient Attention")
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise fallback to CPU

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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Splitting on paragraphs > lines > spaces > characters
    )
    
    chunks = text_splitter.split_documents(docs_path)

    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_embeddings_vector(chunks, vector_db_path="db/chroma"):
    print("Creating embeddings for the chunks...")

    # Using HuggingFaceEmbeddings instead of OpenAIEmbeddings for better performance and flexibility (offering multiple models and offline use)
    embedding_model = HuggingFaceEmbeddings( 
        model_name="BAAI/bge-small-en-v1.5",  # good balance of speed/quality for English text
        model_kwargs={"device": DEVICE},        # change to "cuda" if NVIDIA GPU available; For ROCm check PyTorch ROCm support (rocm.7.2.1 and above)
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},  # recommended for BGE models (cosine similarity)
    )

    print("Writing embeddings to the vector database...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=vector_db_path,
        collection_metadata={"hnsw:space": "cosine"}  # optional metadata for the collection
    )

    return vector_db

def main():
    if DEVICE == "cuda":
        print("Using GPU for embeddings.")
    else:
        print("Using CPU for embeddings.")

    # 1. Load files
    documents = load_docs(docs_path="docs")

    # 2. Chunking the files into smaller pieces
    if documents:
        chunks = chunk_documents(documents)
    else:
        print("No documents to process. Exiting.")
        return

    # 3. Create embeddings and Store them in a vector database (Currently using ChromaDB)
    print("Creating embeddings and storing them in the vector database...")
    vector_db = create_embeddings_vector(chunks, vector_db_path="db/chroma")


if __name__ == "__main__":
    main()