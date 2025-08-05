# v2/batch_evaluator.py
import sys
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.models import load_models
from module.retrieval import retrieve_context
from module.generation import generate_final_answer

# --- 설정 ---
QUESTIONS_CSV_PATH = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG/gradio/questions.csv'
IMAGE_BASE_PATH = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG/images'
OUTPUT_CSV_PATH = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG/v2/evaluation_results.csv'

def main():
    """
    questions.csv를 기반으로 RAG 파이프라인을 일괄 실행하고,
    모든 중간 과정과 최종 결과를 CSV 파일로 저장합니다.
    """
    print("--- V2 배치 평가 시작 ---")
    
    # 1. 모든 모델 로드
    print("1. 모든 모델 로드 중...")
    models = load_models()
    print("-" * 30)

    # 2. 평가할 질문 목록 로드
    print(f"2. 평가 데이터 로드 중... ({QUESTIONS_CSV_PATH})")
    try:
        eval_df = pd.read_csv(QUESTIONS_CSV_PATH)
    except FileNotFoundError:
        print(f"🚨 평가 파일({QUESTIONS_CSV_PATH})을 찾을 수 없습니다.")
        return
    print(f"✅ 총 {len(eval_df)}개의 질문으로 평가를 시작합니다.")
    print("-" * 30)

    results = []

    # 3. 각 질문에 대해 파이프라인 실행
    for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0], desc="[Batch Evaluation]"):
        image_name = row['image_name']
        question = row['question']
        image_path = os.path.join(IMAGE_BASE_PATH, image_name)

        print(f"\n--- [평가 {index+1}/{len(eval_df)}] ---")
        print(f"이미지: {image_path}")
        print(f"질문: {question}")

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"⚠️ 이미지를 찾을 수 없어 건너<binary data, 1 bytes>니다: {image_path}")
            continue

        # 검색 단계 실행 및 중간 결과 저장
        retrieval_results = retrieve_context(
            image=image,
            user_question=question,
            models=models
        )

        # 생성 단계 실행 및 최종 결과 저장
        final_answer, reasoning = generate_final_answer(
            image=image,
            user_question=question,
            image_description=retrieval_results["image_description"],
            context=retrieval_results["retrieved_context"],
            vlm_model=models["vlm_model"],
            text_tokenizer=models["vlm_text_tokenizer"],
            vis_tokenizer=models["vlm_vis_tokenizer"]
        )
        
        # 모든 결과를 하나의 딕셔너리로 취합
        result_row = {
            "image_path": image_path,
            "question": question,
            "keywords": ", ".join(retrieval_results["keywords"]),
            "final_answer": final_answer,
            "reasoning": reasoning,
            "image_description": retrieval_results["image_description"],
            "retrieved_context": retrieval_results["retrieved_context"],
            "search_queries_count": len(retrieval_results["search_queries"]),
        }
        results.append(result_row)

    # 4. 최종 결과를 CSV 파일로 저장
    print("\n--- 평가 완료 ---")
    print(f"4. 최종 결과를 CSV 파일로 저장 중... ({OUTPUT_CSV_PATH})")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료! 총 {len(results_df)}개의 결과가 저장되었습니다.")

if __name__ == "__main__":
    main()
