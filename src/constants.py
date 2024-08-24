import os
from typing import Literal
# light weight faster but less performance using below model
# EMBEDDING_MODEL_NAME="sentence-transformers/multi-qa-MiniLM-L6-cos-v1" 
# EMBEDDING_MODEL_NAME="sentence-transformers/distiluse-base-multilingual-cased-v2" 
EMBEDDING_MODEL_NAME="sentence-transformers/multi-qa-distilbert-cos-v1"
EMBEDDING_MODEL_TYPE:Literal["openai","huggingface"]="openai"
SENTENCE_TRANSFORMERS_HOME="embedding_model"
OPEN_AI_KEY=os.environ.get("OPENAI_API_KEY")
DATA_FOLDER="data"
LOG_FOLDER="log_files"
DB_FOLDER="db"
COLLECTION_NAME="general"
RECREATE_COLLECTION_FOLDER=True

#below is added to avoid deadlock while generating embedding
os.environ["TOKENIZERS_PARALLELISM"] = "false"
