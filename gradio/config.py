# config.py

# 사용할 임베딩 모델 이름
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'

# VLM(시각 언어 모델) 경로
VLM_MODEL_PATH = "AIDC-AI/Ovis2-8B"

# ChromaDB를 저장할 디렉터리 경로
PERSIST_DIRECTORY = '../chroma_db'


# 참고할 모든 컬렉션 이름을 리스트로 관리
COLLECTION_NAMES = [
    "korean_knowledge_base_v2",
    "subway_multilang_all_lines"
]