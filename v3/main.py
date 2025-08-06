# v2/main.py
import sys
import os
from PIL import Image

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_args
from module.models import load_models
from module.retrieval import retrieve_context
from module.generation import generate_final_answer

def main():
    """
    V2 하이브리드 RAG 파이프라인 메인 실행 함수
    """
    args = get_args()
    print("--- V2 하이브리드 RAG 파이프라인 시작 ---")
    print(f"이미지: {args.image_path}")
    print(f"질문: {args.question}")
    print("-" * 30)

    # 1. 모든 모델 로드 (VLM, LLM, Embedding)
    models = load_models()

    # 2. 이미지 로드
    try:
        image = Image.open(args.image_path).convert('RGB')
    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {args.image_path}")
        return

    # 3. 검색 (Retrieval): LLM으로 키워드 추출 후 VDB 검색
    context, image_description = retrieve_context(
        image=image,
        user_question=args.question,
        models=models
    )

    # 4. 생성 (Generation): VLM으로 최종 답변 생성
    final_answer, reasoning = generate_final_answer(
        image=image,
        user_question=args.question,
        image_description=image_description,
        context=context,
        vlm_model=models["vlm_model"],
        text_tokenizer=models["vlm_text_tokenizer"],
        vis_tokenizer=models["vlm_vis_tokenizer"]
    )

    # 5. 결과 출력
    print("\n" + "="*50)
    print("              V2 최종 결과 (하이브리드)")
    print("="*50)
    print(f"\n[질문]\n{args.question}\n")
    print(f"[최종 답변]\n{final_answer}\n")
    print(f"[근거]\n{reasoning}")
    print("="*50)

if __name__ == "__main__":
    main()