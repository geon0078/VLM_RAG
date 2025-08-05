
# v1/config.py
import argparse

# --- 모델 및 경로 설정 ---
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
VLM_MODEL_PATH = "AIDC-AI/Ovis2-8B"
PERSIST_DIRECTORY = '/home/aisw/Project/UST-ETRI-2025/chroma_db'
COLLECTION_NAMES = [
    "korean_knowledge_base_v2",
    "subway_multilang_all_lines"
]

def get_args():
    """커맨드 라인 인자를 파싱하여 반환합니다."""
    parser = argparse.ArgumentParser(description="VLM RAG 파이프라인")
    parser.add_argument(
        "--image_path", 
        type=str, 
        default='/home/aisw/Project/UST-ETRI-2025/VLM_RAG/images/seoul_station.jpg', 
        help="이미지 파일 경로"
    )
    parser.add_argument(
        "--question", 
        type=str, 
        default='이 역의 이름은 무엇인가요?', 
        help="사용자 질문"
    )
    return parser.parse_args()
