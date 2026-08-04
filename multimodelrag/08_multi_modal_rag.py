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


