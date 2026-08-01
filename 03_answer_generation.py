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


#  Search query for testing the retrieval pipeline
query = "What was Microsoft's first hardware product release?" 
 
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # use similarity search with a minimum relevance bar
    search_kwargs={
        "k": 5,                  # retrieve top 5 documents
        "score_threshold": 0.3,  # ...above a similarity score of 0.3
    },
)
 
relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
 
# Load an open, instruction-tuned HuggingFace chat model.
# Qwen2.5-7B-Instruct: ungated (no HF access request needed), strong instruction
# following, and comfortably fits RX 7900 XT in fp16. Swap model_id for any
# other instruct/chat model if you prefer (e.g. "mistralai/Mistral-7B-Instruct-v0.3",
# which is gated and requires accepting terms on huggingface.co first).
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    device_map="auto",
    model_kwargs={"dtype": torch.float16},          # load weights in fp16 to fit VRAM
    pipeline_kwargs={"temperature": 0.2, "max_new_tokens": 512, "do_sample": True},
)
 
# ChatHuggingFace applies the model's chat template (system/user turns) on top of the
# raw text-generation pipeline, and returns a proper AIMessage with .content — a plain
# HuggingFacePipeline expects a raw string prompt, not chat messages.
chat_model = ChatHuggingFace(llm=llm)
 
# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
