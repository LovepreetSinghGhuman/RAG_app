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

def main():
    if DEVICE == "cuda":
        print("Using GPU for embeddings.")
    else:
        print("Using CPU for embeddings.")

    # 1. Load files


if __name__ == "__main__":
    main()