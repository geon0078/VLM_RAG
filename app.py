import gradio as gr
import os
import sys
import torch
from PIL import Image
import time
import io
import contextlib

# --- 프로젝트 경로 설정 ---
base_dir = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG'
sys.path.insert(0, base_dir)

# --- V1, V3 모듈에서 필요한 함수들을 명확한 경로로 직접 임포트 ---
try:
    from v1.module.models import load_all_models as load_v1_models
    from v1.module.pipeline import run_rag_pipeline as run_v1_pipeline

    from v3.module.models import load_models as load_v3_models
    from v3.module.retrieval import retrieve_context as retrieve_v3_context
    from v3.module.generation import generate_final_answer as generate_v3_answer
except ImportError as e:
    print(f"모듈 임포트 중 오류 발생: {e}")
    print("VLM_RAG 디렉토리 구조를 확인해주세요. v1과 v3 폴더가 존재해야 합니다.")
    sys.exit(1)

# --- 모델을 전역 변수로 선언 ---
V1_MODELS = None
V3_MODELS = None

def load_models_globally():
    """
    Gradio 앱 시작 시 모든 모델을 한 번만 로드하여 전역 변수에 저장합니다.
    """
    global V1_MODELS, V3_MODELS
    
    print("="*50)
    print("Gradio 앱 시작... 모든 모델을 로드합니다.")
    print("이 과정은 몇 분 정도 소요될 수 있습니다. 잠시만 기다려주세요.")
    print("="*50)

    # stdout을 리디렉션하여 모델 로딩 로그를 UI에 표시할 수도 있지만,
    # 복잡성을 줄이기 위해 터미널에만 표시합니다.
    print("\n[1/2] V1 모델 로딩 중...")
    vlm_model_v1, text_tokenizer_v1, vis_tokenizer_v1, embedding_model_v1, collections_v1 = load_v1_models()
    V1_MODELS = (vlm_model_v1, text_tokenizer_v1, vis_tokenizer_v1, embedding_model_v1, collections_v1)
    print("✅ V1 모델 로드 완료.")

    print("\n[2/2] V3 모델 로딩 중...")
    V3_MODELS = load_v3_models()
    print("✅ V3 모델 로드 완료.")
    
    print("\n🎉 모든 모델 로드가 완료되었습니다. Gradio 앱을 시작합니다.")
    print("="*50)

# --- Gradio 앱을 위한 실시간 스트리밍 함수 ---

def compare_v1_and_v3_stream(image, question):
    """
    Gradio 스트리밍을 위한 생성기 함수.
    모델 실행 단계별로 로그를 캡처하고 UI에 실시간으로 yield합니다.
    """
    if image is None or question is None or question.strip() == "":
        yield {
            v1_answer_output: "이미지를 업로드하고 질문을 입력해주세요.",
            v1_reasoning_output: "",
            v1_debug_output: "",
            v3_answer_output: "이미지를 업로드하고 질문을 입력해주세요.",
            v3_reasoning_output: "",
            v3_debug_output: "",
        }
        return

    # UI 초기화
    yield {
        v1_answer_output: "V1 모델 실행 중...",
        v1_reasoning_output: "",
        v1_debug_output: "",
        v3_answer_output: "V3 모델 실행 중...",
        v3_reasoning_output: "",
        v3_debug_output: "",
    }

    v1_log_stream = io.StringIO()
    v3_log_stream = io.StringIO()
    
    # --- V1 추론 실행 및 실시간 로그 출력 ---
    v1_answer, v1_reasoning = "", ""
    try:
        with contextlib.redirect_stdout(v1_log_stream):
            vlm_model_v1, text_tokenizer_v1, vis_tokenizer_v1, embedding_model_v1, collections_v1 = V1_MODELS
            v1_answer, v1_reasoning = run_v1_pipeline(
                image=image,
                user_question=question,
                vlm_model=vlm_model_v1,
                text_tokenizer=text_tokenizer_v1,
                vis_tokenizer=vis_tokenizer_v1,
                embedding_model=embedding_model_v1,
                chroma_collections=collections_v1
            )
        v1_debug_log = v1_log_stream.getvalue()
        yield { v1_debug_output: v1_debug_log, v1_answer_output: "V1 모델 완료" }
    except Exception as e:
        v1_debug_log = f"V1 실행 중 오류 발생:\n{v1_log_stream.getvalue()}\n\n{e}"
        yield { v1_debug_output: v1_debug_log, v1_answer_output: "오류 발생" }


    # --- V3 추론 실행 및 실시간 로그 출력 ---
    v3_answer, v3_reasoning = "", ""
    try:
        with contextlib.redirect_stdout(v3_log_stream):
            retrieval_results = retrieve_v3_context(
                image=image,
                user_question=question,
                models=V3_MODELS
            )
            # 중간 로그를 UI에 업데이트
            yield { v3_debug_output: v3_log_stream.getvalue() }

            v3_answer, v3_reasoning = generate_v3_answer(
                image=image,
                user_question=question,
                image_description=retrieval_results["image_description"],
                context=retrieval_results["retrieved_context"],
                vlm_model=V3_MODELS["vlm_model"],
                text_tokenizer=V3_MODELS["vlm_text_tokenizer"],
                vis_tokenizer=V3_MODELS["vlm_vis_tokenizer"]
            )
        v3_debug_log = v3_log_stream.getvalue()
        yield { v3_debug_output: v3_debug_log, v3_answer_output: "V3 모델 완료" }
    except Exception as e:
        v3_debug_log = f"V3 실행 중 오류 발생:\n{v3_log_stream.getvalue()}\n\n{e}"
        yield { v3_debug_output: v3_debug_log, v3_answer_output: "오류 발생" }

    # --- 최종 결과 UI에 표시 ---
    yield {
        v1_answer_output: v1_answer,
        v1_reasoning_output: v1_reasoning,
        v1_debug_output: v1_debug_log,
        v3_answer_output: v3_answer,
        v3_reasoning_output: v3_reasoning,
        v3_debug_output: v3_debug_log,
    }


# --- Gradio UI 인터페이스 정의 ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚇 V1 vs V3 모델 아키텍처 비교 (실시간 로그)")
    gr.Markdown("모델이 모두 로드된 후 이미지를 업로드하고 질문을 입력하면, V1과 V3 모델의 실행 로그와 최종 결과를 확인할 수 있습니다.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="이미지 업로드")
            question_input = gr.Textbox(label="질문 입력", placeholder="예: 이 역의 이름은 무엇인가요?")
            submit_btn = gr.Button("결과 비교하기", variant="primary")
            
            image_dir = os.path.join(base_dir, "images")
            try:
                all_files = os.listdir(image_dir)
                image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
                example_images = [os.path.join(image_dir, f) for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]
                
                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=image_input,
                        label="예시 이미지"
                    )
            except FileNotFoundError:
                gr.Markdown(f"⚠️ 예시 이미지 디렉토리를 찾을 수 없습니다: `{image_dir}`")

        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## V1 모델 결과")
                    v1_answer_output = gr.Textbox(label="V1 최종 답변", lines=5, interactive=False)
                    v1_reasoning_output = gr.Textbox(label="V1 근거 및 참고 문서", lines=10, interactive=False)
                    with gr.Accordion("V1 디버그 로그 보기", open=True):
                        v1_debug_output = gr.Textbox(label="V1 실시간 출력 로그", lines=15, interactive=False)

                with gr.Column():
                    gr.Markdown("## V3 모델 결과")
                    v3_answer_output = gr.Textbox(label="V3 최종 답변", lines=5, interactive=False)
                    v3_reasoning_output = gr.Textbox(label="V3 근거 및 참고 문서 (점수 포함)", lines=10, interactive=False)
                    with gr.Accordion("V3 디버그 로그 보기", open=True):
                        v3_debug_output = gr.Textbox(label="V3 실시간 출력 로그", lines=15, interactive=False)

    # 버튼 클릭 이벤트를 스트리밍 함수에 연결
    submit_btn.click(
        fn=compare_v1_and_v3_stream,
        inputs=[image_input, question_input],
        outputs=[v1_answer_output, v1_reasoning_output, v1_debug_output, v3_answer_output, v3_reasoning_output, v3_debug_output]
    )

if __name__ == "__main__":
    # 1. 앱 시작 시 모델을 전역적으로 로드
    load_models_globally()
    
    # 2. 모델 로드가 완료된 후 Gradio 앱 실행
    demo.launch(share=True)
