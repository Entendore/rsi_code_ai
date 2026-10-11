import os

class Config:
    # API Keys
    E2B_API_KEY = os.getenv("E2B_API_KEY")
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Custom Code Model Parameters
    VOCAB_SIZE = 500
    MAX_SEQ_LEN = 128      
    EMBED_DIM = 128        
    NUM_HEADS = 4          
    FF_DIM = 256           
    BEAM_WIDTH = 3         
    
    # RSI & Learning Parameters
    MAX_RSI_RECURSION_DEPTH = 5
    MAX_DEBUG_RETRIES = 2
    MAX_TOKENS_PER_CYCLE = 500_000
    MIN_IMPROVEMENT_THRESHOLD = 0.05
    FINE_TUNE_LEARNING_RATE = 1e-3
    RL_LEARNING_RATE = 1e-4
    DPO_LEARNING_RATE = 5e-5
    EWC_LAMBDA = 0.001
    GRADIENT_CLIP_NORM = 1.0
    
    # Checkpointing & Memory
    CHECKPOINT_PATH = "./ai_model_weights/CustomCodeTransformer"
    CHROMA_DB_PATH = "./ai_memory_db"
