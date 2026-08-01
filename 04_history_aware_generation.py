import os
import warnings
 
import torch
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
 
warnings.filterwarnings("ignore")
load_dotenv()


# SETUP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise fallback to CPU
DBPATH = os.getenv("VECTOR_DB_PATH", "db/chroma")  # Default path env or "db/chroma"
 
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
    pipeline_kwargs={"temperature": 0.2, "max_new_tokens": 512, "do_sample": True},
)

chat_history = []  # Initialize an empty list to store the conversation history

def start_chat():
    print("Welcome to the RAG Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break
        
        # Retrieve relevant documents based on user input
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.3,
            },
        )
        relevant_docs = retriever.invoke(user_input)
        
        # Combine retrieved documents into a single context string
        combined_context = "\n".join(f"- {doc.page_content}" for doc in relevant_docs)
        
        # Create a prompt for the LLM
        prompt = f"""Based on the following documents, answer the question: {user_input}

Documents:
{combined_context}

Please provide a concise and accurate answer based on the information from the documents. If the answer is not present in the documents, respond with "I don't know."
"""

if __name__ == "__main__":
    start_chat()