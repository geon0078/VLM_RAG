
# v1/module/models.py
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
from ..config import VLM_MODEL_PATH, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, COLLECTION_NAMES

def load_all_models():
    """
    애플리케이션에 필요한 모든 모델과 DB 커넥션을 로드합니다.
    """
    print("VLM 모델을 로드하는 중...")
    vlm_model = AutoModelForCausalLM.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    text_tokenizer = vlm_model.get_text_tokenizer()
    vis_tokenizer = vlm_model.get_visual_tokenizer()
    print("✅ VLM 모델 로드 완료.")

    print("임베딩 모델과 ChromaDB를 로드하는 중...")
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

    return vlm_model, text_tokenizer, vis_tokenizer, embedding_model, collections
