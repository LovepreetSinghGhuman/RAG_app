import os
import warnings
import importlib
from langchain_text_splitters import CharacterTextSplitter
import torch


from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
 
warnings.filterwarnings("ignore")
load_dotenv()

module = importlib.import_module("01_ingestion_pipline")
load_docs = module.load_docs
create_embeddings_vector = module.create_embeddings_vector

# SETUP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise fallback to CPU
DBPATH = os.getenv("VECTOR_DB_PATH", "db/chroma")  # Default path env or "db/chroma"


# 5 types of chunking/splitting strategies: "Recursive", "Character", "Document-Specific", "Semantic", "Agentic"

# IMPORTANT: model_name and encode_kwargs must match ingestion_pipline.py exactly.
# A mismatch here silently produces a different vector space than the one your documents were embedded into, which breaks similarity search.
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # must match ingestion_pipline.py
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},  # must match ingestion_pipline.py
)
 
db = Chroma(
    persist_directory=DBPATH,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# Setup AI model
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

# ChatHuggingFace applies the model's chat template and returns a proper AIMessage
# with .content — invoking a raw HuggingFacePipeline with chat messages does not do this.
chat_model = ChatHuggingFace(llm=llm)


def chunk_documents_character(docs_path, chunk_size=800, chunk_overlap=0):
    print("Chunking documents into smaller pieces ...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Splitting on paragraphs > lines > spaces > characters
    )
    
    chunks = text_splitter.split_documents(docs_path)

    print(f"Total chunks created: {len(chunks)}")
    return chunks

def main():
    if DEVICE == "cuda":
        print("Using GPU for embeddings.")
    else:
        print("Using CPU for embeddings.")

    # 1. Load files
    documents = load_docs(docs_path="docs")

    # 2. Chunking the files into smaller pieces
    if documents:
        chunks = chunk_documents_character(documents)
    else:
        print("No documents to process. Exiting.")
        return

    # 3. Create embeddings and Store them in a vector database (Currently using ChromaDB)
    print("Creating embeddings and storing them in the vector database...")
    vector_db = create_embeddings_vector(chunks, vector_db_path="db/chroma")


if __name__ == "__main__":
    main()