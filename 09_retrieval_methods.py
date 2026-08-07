import os
import warnings
import importlib
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

# --- Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else fallback to CPU
DB_PATH = os.getenv("VECTOR_DB_PATH", "db/chroma")  # Vector DB storage path, overridable via .env

# Tesla text to chunk
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# Create the prompt
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

# 5 types of chunking/splitting strategies: "Recursive", "Character", "Document-Specific", "Semantic", "Agentic"
# Here we are implementing the "Agentic" chunking strategy, which splits documents into smaller pieces based on agent-driven criteria (ie. semantic relevance via prompting a llm).

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

chatmodel = ChatHuggingFace(llm=llm)

# --- Main ---
print(f"Using {'GPU' if DEVICE == 'cuda' else 'CPU'} for compute.")

# 1. Send prompt and text to the LLM for agentic chunking
response = llm.invoke([HumanMessage(content=prompt)])
marked_text = response # no .content needed since we are using a text-generation model, not a chat model

# 2. Split the text into chunks based on the "<<<SPLIT>>>" markers
chunks = marked_text.split("<<<SPLIT>>>")

# 3. Clean up the chunks by stripping whitespace and removing empty chunks
cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

# 4. Display the resulting chunks
print(f"Agentic chunking produced {len(cleaned_chunks)} chunks:")
for i, chunk in enumerate(cleaned_chunks, 1):
    print(f"  Chunk {i}: {chunk}...")  # Show first 60 characters of each chunk



