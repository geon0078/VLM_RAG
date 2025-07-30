# app.py

import gradio as gr
from models import load_all_models
from pipeline import run_rag_pipeline

# --- 1. 애플리케이션 시작 시 모델 로딩 ---
print(" 애플리케이션 시작... 모든 모델 로딩을 시작합니다. ")
try:
    VLM, TXT_TOKENIZER, VIS_TOKENIZER, EMB_MODEL, DB_COLLECTION = load_all_models()
    print("✨ 모든 모델과 DB가 준비되었습니다. Gradio UI를 시작합니다. ✨")
    MODELS_LOADED = True
except Exception as e:
    print(f"🚨 치명적 오류: 모델 로딩에 실패했습니다. {e}")
    MODELS_LOADED = False

# --- 2. Gradio 인터페이스를 위한 래퍼(Wrapper) 함수 ---
def gradio_interface_func(image, user_question, progress=gr.Progress(track_tqdm=True)):
    """Gradio의 click 이벤트에 연결될 함수"""
    if not MODELS_LOADED:
        raise gr.Error("모델이 정상적으로 로드되지 않았습니다. 서버를 재시작하고 로그를 확인해주세요.")
    
    # 실제 파이프라인 함수에 로드된 모델 객체들을 전달
    return run_rag_pipeline(
        image, user_question, VLM, TXT_TOKENIZER, VIS_TOKENIZER, EMB_MODEL, DB_COLLECTION, progress
    )

# --- 3. Gradio UI 구성 ---
with gr.Blocks(theme=gr.themes.Soft(), title="Vision RAG") as demo:
    gr.Markdown("# Vision RAG: 이미지와 지식베이스(DB)를 활용한 Q&A 🤖")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. 입력")
            image_input = gr.Image(type="pil", label="이미지 업로드")
            question_input = gr.Textbox(label="질문 입력", placeholder="이미지에 대해 질문하세요...")
            submit_button = gr.Button("답변 생성하기", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## 2. 결과")
            answer_output = gr.Textbox(label="최종 답변", interactive=False, lines=5)
            with gr.Accordion("상세 과정 보기 (디버깅용)", open=False):
                desc_output = gr.Textbox(label="(1) 이미지 분석 결과 (생성된 설명)", interactive=False, lines=5)
                context_output = gr.Textbox(label="(2) 지식베이스 검색 결과 (검색된 문맥)", interactive=False, lines=8)
    
    if MODELS_LOADED:
        # 버튼 클릭 이벤트 연결
        submit_button.click(
            fn=gradio_interface_func,
            inputs=[image_input, question_input],
            outputs=[answer_output, desc_output, context_output]
        )
        
        # 예시 제공
        gr.Examples(
            examples=[["./data/Daejeon_station.jpeg", "대전역의 역사 관리실 전화번호는?"]],
            inputs=[image_input, question_input],
            outputs=[answer_output, desc_output, context_output],
            fn=gradio_interface_func,
            cache_examples=False
        )
    else:
        gr.Markdown("## 🚨 모델 로딩 실패! 🚨")
        gr.Markdown("애플리케이션을 사용할 수 없습니다. 터미널의 오류 메시지를 확인하고 서버를 재시작해주세요.")


if __name__ == "__main__":
    demo.launch()