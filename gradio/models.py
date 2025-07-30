# models.py

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
from config import VLM_MODEL_PATH, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, COLLECTION_NAME

def load_all_models():
    """
    애플리케이션에 필요한 모든 모델과 DB 커넥션을 로드합니다.
    """
    print("Loading VLM model...")
    vlm_model = AutoModelForCausalLM.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    text_tokenizer = vlm_model.get_text_tokenizer()
    vis_tokenizer = vlm_model.get_visual_tokenizer()
    print("✅ VLM model loaded.")

    print("Loading embedding model and ChromaDB...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    try:
        chroma_collection = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ ChromaDB collection '{COLLECTION_NAME}' loaded with {chroma_collection.count()} documents.")
    except Exception as e:
        print(f"🚨 ChromaDB 컬렉션 로드 실패: {e}")
        chroma_collection = None
    print("✅ Embedding model and ChromaDB loaded.")

    return vlm_model, text_tokenizer, vis_tokenizer, embedding_model, chroma_collection