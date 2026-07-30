import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_docs(docs_path="docs"):
    print("Loading documents from the directory...")

    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        print(f"Data directory not found: {docs_path}")
        return
    
    # Load all text files from the specified directory
    loader = DirectoryLoader(
        docs_path, 
        glob="**/*.txt", 
        loader_cls=TextLoader
    )

    docs = loader.load()

    if len(docs) == 0:
        print(f"No text files found in the directory: {docs_path}")
        return
    
    for i, doc in enumerate(docs):
        print(f"Document {i+1}: {doc.metadata.get('source', 'Unknown source')} - {len(doc.page_content)} characters")

    print("Embeddings created and stored in ChromaDB.")

def main():
    print("Starting ingestion pipeline...")
    # 1. Load files
    documents = load_docs(docs_path="docs")
    # 2. Chuncking the files into smaller pieces

    # 3. Create embeddings and Store them in a vector database (Currently using ChromaDB)


if __name__ == "__main__":
    main()
