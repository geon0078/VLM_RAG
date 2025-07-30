# app2.py


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 두 번째 GPU만 사용

import gradio as gr
from models import load_all_models
from pipeline import run_rag_pipeline, run_vlm_only_pipeline
from PIL import Image

# --- 1. 애플리케이션 시작 시 모델 로딩 ---
print(" 비교 앱 시작... 모든 모델 로딩을 시작합니다. ")
try:
    VLM, TXT_TOKENIZER, VIS_TOKENIZER, EMB_MODEL, DB_COLLECTION = load_all_models()
    print("✨ 모든 모델과 DB가 준비되었습니다. Gradio UI를 시작합니다. ✨")
    MODELS_LOADED = True
except Exception as e:
    print(f"🚨 치명적 오류: 모델 로딩에 실패했습니다. {e}")
    MODELS_LOADED = False

# --- 2. Gradio 인터페이스를 위한 래퍼(Wrapper) 함수 ---
def gradio_comparison_interface(image, user_question, progress=gr.Progress(track_tqdm=True)):
    """VLM 단독 답변과 RAG 적용 답변을 모두 생성하여 반환합니다."""
    if not MODELS_LOADED:
        raise gr.Error("모델이 정상적으로 로드되지 않았습니다. 서버를 재시작하고 로그를 확인해주세요.")
    
    progress(0, desc="VLM 단독 답변 생성 중...")
    # [수정] 이제 vlm_only_pipeline은 (답변, 근거) 2개를 반환합니다.
    vlm_answer, vlm_reasoning = run_vlm_only_pipeline(image, user_question, VLM, TXT_TOKENIZER, VIS_TOKENIZER)
    
    # RAG 적용 파이프라인 실행
    rag_answer, rag_desc, rag_context = run_rag_pipeline(
        image, user_question, VLM, TXT_TOKENIZER, VIS_TOKENIZER, EMB_MODEL, DB_COLLECTION, progress
    )
    
    # [수정] UI 컴포넌트 순서에 맞게 모든 결과를 반환
    return vlm_answer, vlm_reasoning, rag_answer, rag_desc, rag_context

# --- 3. Gradio UI 구성 ---
with gr.Blocks(theme=gr.themes.Soft(), title="VLM vs RAG-VLM") as demo:
    gr.Markdown("# VLM vs RAG-VLM 비교 분석 🤖")
    gr.Markdown("동일한 이미지와 질문에 대해 VLM 단독 답변과 RAG 적용 답변을 비교합니다.")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. 입력")
            image_input = gr.Image(type="pil", label="이미지 업로드")
            question_input = gr.Textbox(label="질문 입력", placeholder="이미지에 대해 질문하세요...")
            submit_button = gr.Button("결과 비교하기", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("## 2. VLM 단독 답변")
            # [수정] lines와 max_lines를 함께 지정하여 스크롤바 생성 제어
            vlm_answer_output = gr.Textbox(
                label="모델의 자체 지식 기반 답변", interactive=False, lines=10, max_lines=15
            )
            with gr.Accordion("답변 근거 보기 (VLM)", open=False):
                # [수정] lines와 max_lines를 함께 지정
                vlm_reasoning_output = gr.Textbox(
                    label="VLM의 추론 과정", interactive=False, lines=10, max_lines=15
                )

        with gr.Column(scale=1):
            gr.Markdown("## 3. RAG 적용 답변")
            # [수정] lines와 max_lines를 함께 지정
            rag_answer_output = gr.Textbox(
                label="지식베이스(DB) 참고 답변", interactive=False, lines=10, max_lines=15
            )
            with gr.Accordion("상세 과정 보기 (RAG)", open=False):
                # [수정] lines와 max_lines를 함께 지정
                rag_desc_output = gr.Textbox(
                    label="(1) 이미지 분석 결과", interactive=False, lines=10, max_lines=15
                )
                # [수정] lines와 max_lines를 함께 지정
                rag_context_output = gr.Textbox(
                    label="(2) 지식베이스 검색 결과", interactive=False, lines=10, max_lines=15
                )

    if MODELS_LOADED:
        submit_button.click(
            fn=gradio_comparison_interface,
            inputs=[image_input, question_input],
            outputs=[vlm_answer_output, vlm_reasoning_output, rag_answer_output, rag_desc_output, rag_context_output]
        )
    else:
        gr.Markdown("## 🚨 모델 로딩 실패! 🚨 애플리케이션을 사용할 수 없습니다.")

if __name__ == "__main__":
    demo.launch()