# v1/main.py
import sys
import os
from PIL import Image

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_args
from module.models import load_all_models
from module.pipeline import run_rag_pipeline

def main():
    """
    메인 실행 함수
    """
    args = get_args()

    # 1. 모델 로드
    vlm_model, text_tokenizer, vis_tokenizer, embedding_model, collections = load_all_models()

    # 2. 이미지 로드
    try:
        image = Image.open(args.image_path).convert('RGB')
    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {args.image_path}")
        return

    # 3. RAG 파이프라인 실행
    final_answer, reasoning = run_rag_pipeline(
        image=image,
        user_question=args.question,
        vlm_model=vlm_model,
        text_tokenizer=text_tokenizer,
        vis_tokenizer=vis_tokenizer,
        embedding_model=embedding_model,
        chroma_collections=collections
    )

    # 4. 결과 출력
    print("\n" + "="*50)
    print("                최종 결과")
    print("="*50)
    print(f"\n[질문] {args.question}\n")
    print(f"[최종 답변]\n{final_answer}\n")
    print(f"[근거]\n{reasoning}")
    print("="*50)

if __name__ == "__main__":
    main()