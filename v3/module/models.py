# v2/module/models.py
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..config import VLM_MODEL_PATH, LLM_MODEL_PATH, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, COLLECTION_NAMES

def load_models():
    """
    애플리케이션에 필요한 모든 모델(VLM, LLM, Embedding)과 DB 커넥션을 로드합니다.
    """
    # --- 1. VLM 로드 ---
    print("1. VLM 모델 로드 중... (Ovis)")
    vlm_model = AutoModelForCausalLM.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    vlm_text_tokenizer = vlm_model.get_text_tokenizer()
    vlm_vis_tokenizer = vlm_model.get_visual_tokenizer()
    print("✅ VLM 모델 로드 완료.")

    # --- 2. LLM 로드 (키워드 추출용) ---
    print("2. LLM 모델 로드 중... (Qwen)")
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
    print("✅ LLM 모델 로드 완료.")

    # --- 3. 임베딩 모델 및 ChromaDB 로드 ---
    print("3. 임베딩 모델 및 ChromaDB 로드 중...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    collections = {}
    for cname in COLLECTION_NAMES:
        try:
            cobj = client.get_collection(name=cname)
            print(f"✅ ChromaDB 컬렉션 '{cname}' 로드 완료 ({cobj.count()}개 문서).")
            collections[cname] = cobj
        except Exception as e:
            print(f"🚨 '{cname}' 컬렉션 로드 실패: {e}")
            collections[cname] = None
    print("✅ 임베딩 모델 및 ChromaDB 로드 완료.")

    return {
        "vlm_model": vlm_model,
        "vlm_text_tokenizer": vlm_text_tokenizer,
        "vlm_vis_tokenizer": vlm_vis_tokenizer,
        "llm_model": llm_model,
        "llm_tokenizer": llm_tokenizer,
        "embedding_model": embedding_model,
        "collections": collections
    }