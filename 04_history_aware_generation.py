import os
import torch
import warnings
 
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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

chat_history = []  # Initialize an empty list to store the conversation history

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = chat_model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
    
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = chat_model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation (capped to the last 3 turns to avoid
    # unbounded context growth over a long chat session)
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    del chat_history[:-6]  # keep only the most recent 3 question/answer pairs
    
    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()