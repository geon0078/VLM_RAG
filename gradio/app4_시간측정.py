# app2.py

import os
import logging
import time
from typing import Tuple, Optional, Any, Dict
from contextlib import contextmanager

# GPU 설정을 최상단으로 이동
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 두 번째 GPU만 사용

import gradio as gr
from models import load_all_models
from pipeline import run_rag_pipeline, run_vlm_only_pipeline
from PIL import Image

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    """모델 상태를 관리하는 클래스"""
    def __init__(self):
        self.models_loaded = False
        self.vlm = None
        self.txt_tokenizer = None
        self.vis_tokenizer = None
        self.emb_model = None
        self.collections = None
    
    def load_models(self) -> bool:
        """모든 모델을 로딩합니다."""
        try:
            logger.info("🚀 비교 앱 시작... 모든 모델 로딩을 시작합니다.")
            
            self.vlm, self.txt_tokenizer, self.vis_tokenizer, self.emb_model, self.collections = load_all_models()
            
            logger.info("✨ 모든 모델과 DB가 준비되었습니다. Gradio UI를 시작합니다.")
            self.models_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"🚨 치명적 오류: 모델 로딩에 실패했습니다. {e}")
            self.models_loaded = False
            return False
    
    def is_ready(self) -> bool:
        """모델이 준비되었는지 확인합니다."""
        return self.models_loaded

# 전역 모델 매니저 인스턴스
model_manager = ModelManager()

@contextmanager
def timer(operation_name: str):
    """시간 측정을 위한 컨텍스트 매니저"""
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"{operation_name} 소요 시간: {elapsed_time:.2f}초")

@contextmanager
def error_handler(operation_name: str):
    """에러 처리를 위한 컨텍스트 매니저"""
    try:
        yield
    except Exception as e:
        logger.error(f"{operation_name} 중 오류 발생: {e}")
        raise gr.Error(f"{operation_name} 중 오류가 발생했습니다: {str(e)}")

def format_time(seconds: float) -> str:
    """시간을 사용자 친화적인 형태로 포맷팅"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}초"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}분 {remaining_seconds:.1f}초"

def create_performance_summary(vlm_time: float, rag_time: float) -> str:
    """성능 요약 정보를 생성"""
    total_time = vlm_time + rag_time
    vlm_percentage = (vlm_time / total_time) * 100 if total_time > 0 else 0
    rag_percentage = (rag_time / total_time) * 100 if total_time > 0 else 0
    
    speedup = vlm_time / rag_time if rag_time > 0 else 0
    
    summary = f"""
📊 **성능 분석 요약**

⏱️ **처리 시간**
• VLM 단독: {format_time(vlm_time)} ({vlm_percentage:.1f}%)
• RAG 적용: {format_time(rag_time)} ({rag_percentage:.1f}%)
• 전체 시간: {format_time(total_time)}

🚀 **성능 비교**
"""
    
    if speedup > 1:
        summary += f"• VLM이 RAG보다 {speedup:.1f}배 빠름"
    elif speedup < 1 and speedup > 0:
        summary += f"• RAG가 VLM보다 {1/speedup:.1f}배 빠름"
    else:
        summary += "• 처리 시간 비슷함"
    
    return summary

def validate_inputs(image: Optional[Image.Image], question: str) -> None:
    """입력값 검증"""
    if image is None:
        raise gr.Error("이미지를 업로드해주세요.")
    
    if not question or question.strip() == "":
        raise gr.Error("질문을 입력해주세요.")
    
    if len(question.strip()) < 3:
        raise gr.Error("질문이 너무 짧습니다. 최소 3글자 이상 입력해주세요.")

def gradio_comparison_interface(
    image: Optional[Image.Image], 
    user_question: str, 
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[str, str, str, str, str, str, str]:
    """VLM 단독 답변과 RAG 적용 답변을 모두 생성하여 반환합니다."""
    
    # 모델 상태 확인
    if not model_manager.is_ready():
        raise gr.Error("모델이 정상적으로 로드되지 않았습니다. 서버를 재시작하고 로그를 확인해주세요.")
    
    # 입력값 검증
    validate_inputs(image, user_question)
    
    # 질문 전처리
    user_question = user_question.strip()
    
    # 시간 측정 변수
    vlm_time = 0
    rag_time = 0
    
    try:
        # VLM 단독 답변 생성 (시간 측정)
        with error_handler("VLM 단독 답변 생성"):
            progress(0.1, desc="🧠 VLM 단독 답변 생성 중...")
            
            vlm_start_time = time.time()
            vlm_answer, vlm_reasoning = run_vlm_only_pipeline(
                image, 
                user_question, 
                model_manager.vlm, 
                model_manager.txt_tokenizer, 
                model_manager.vis_tokenizer
            )
            vlm_end_time = time.time()
            vlm_time = vlm_end_time - vlm_start_time
            
            logger.info(f"VLM 처리 완료: {format_time(vlm_time)}")
        
        # RAG 적용 파이프라인 실행 (시간 측정)
        with error_handler("RAG 적용 답변 생성"):
            progress(0.5, desc="📚 RAG 적용 답변 생성 중...")
            
            rag_start_time = time.time()
            rag_answer, rag_desc, rag_context = run_rag_pipeline(
                image, 
                user_question, 
                model_manager.vlm, 
                model_manager.txt_tokenizer, 
                model_manager.vis_tokenizer, 
                model_manager.emb_model, 
                model_manager.collections, 
                progress
            )
            rag_end_time = time.time()
            rag_time = rag_end_time - rag_start_time
            
            logger.info(f"RAG 처리 완료: {format_time(rag_time)}")
        
        progress(1.0, desc="✅ 완료!")
        
        # 결과 검증
        if not vlm_answer or not rag_answer:
            raise gr.Error("답변 생성에 실패했습니다. 다시 시도해주세요.")
        
        # 시간 정보 포맷팅
        vlm_time_str = f"⏱️ 처리 시간: {format_time(vlm_time)}"
        rag_time_str = f"⏱️ 처리 시간: {format_time(rag_time)}"
        
        # 성능 요약 생성
        performance_summary = create_performance_summary(vlm_time, rag_time)
        
        return (
            vlm_answer, 
            vlm_reasoning, 
            rag_answer, 
            rag_desc, 
            rag_context,
            vlm_time_str,
            rag_time_str,
            performance_summary
        )
        
    except gr.Error:
        # Gradio 에러는 그대로 재발생
        raise
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}")
        raise gr.Error(f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}")

def create_textbox_config(label: str, lines: int = 8, scrollable: bool = True) -> dict:
    """텍스트박스 공통 설정을 반환합니다."""
    config = {
        "label": label,
        "interactive": False,
        "show_copy_button": True,
        "show_label": True,
    }
    
    if scrollable:
        # 스크롤 가능한 텍스트박스 설정
        config.update({
            "lines": lines,
            "max_lines": lines,  # lines와 동일하게 설정하여 고정 높이
            "autoscroll": False  # 자동 스크롤 비활성화로 사용자가 직접 제어
        })
    else:
        # 자동 크기 조절 텍스트박스
        config.update({
            "lines": lines,
            "autoscroll": True
        })
    
    return config

def create_demo() -> gr.Blocks:
    """Gradio 데모 인터페이스를 생성합니다."""
    
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        title="VLM vs RAG-VLM",
        css="""
        .container { max-width: 1400px; margin: auto; }
        .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        
        /* 텍스트박스 스크롤 강제 적용 */
        .scroll-area textarea {
            overflow-y: auto !important;
            resize: vertical !important;
            max-height: 300px !important;
            height: 200px !important;
        }
        
        /* Gradio 텍스트박스 스타일 개선 */
        .gr-textbox textarea {
            overflow-y: auto !important;
            resize: vertical !important;
        }
        
        /* 아코디언 내부 텍스트박스 */
        .gr-accordion .gr-textbox textarea {
            max-height: 200px !important;
            height: 150px !important;
            overflow-y: auto !important;
        }
        
        /* 시간 표시 스타일 */
        .time-display input {
            background: linear-gradient(90deg, #4CAF50, #45a049) !important;
            color: white !important;
            font-weight: bold !important;
            text-align: center !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px !important;
            margin: 5px 0 !important;
            font-size: 14px !important;
        }
        
        /* 성능 요약 스타일 */
        .performance-summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            padding: 20px !important;
            border-radius: 12px !important;
            margin: 10px 0 !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        }
        
        .performance-summary h3, .performance-summary h4 {
            color: white !important;
            margin-top: 0 !important;
        }
        """
    ) as demo:
        
        gr.Markdown("# 🤖 VLM vs RAG-VLM 비교 분석")
        gr.Markdown("동일한 이미지와 질문에 대해 VLM 단독 답변과 RAG 적용 답변을 비교합니다.")
        
        if not model_manager.is_ready():
            gr.Markdown("## 🚨 모델 로딩 실패!")
            gr.Markdown("애플리케이션을 사용할 수 없습니다. 서버를 재시작해주세요.")
            return demo
        
        gr.Markdown("---")

        with gr.Row():
            # 입력 섹션
            with gr.Column(scale=1):
                gr.Markdown("## 📤 입력")
                image_input = gr.Image(
                    type="pil", 
                    label="이미지 업로드",
                    height=300
                )
                question_input = gr.Textbox(
                    label="질문 입력", 
                    placeholder="이미지에 대해 구체적으로 질문하세요...",
                    lines=3
                )
                
                with gr.Row():
                    submit_button = gr.Button(
                        "🔍 결과 비교하기", 
                        variant="primary",
                        size="lg"
                    )
                    clear_button = gr.Button(
                        "🗑️ 초기화", 
                        variant="secondary"
                    )
            
            # VLM 단독 답변 섹션
            with gr.Column(scale=1):
                gr.Markdown("## 🧠 VLM 단독 답변")
                vlm_answer_output = gr.Textbox(
                    **create_textbox_config("모델의 자체 지식 기반 답변", lines=10, scrollable=True),
                    elem_classes=["scroll-area"]
                )
                
                # VLM 시간 표시
                vlm_time_output = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_classes=["time-display"]
                )
                
                with gr.Accordion("💭 답변 근거 보기", open=False):
                    vlm_reasoning_output = gr.Textbox(
                        **create_textbox_config("VLM의 추론 과정", lines=8, scrollable=True),
                        elem_classes=["scroll-area"]
                    )

            # RAG 적용 답변 섹션
            with gr.Column(scale=1):
                gr.Markdown("## 📚 RAG 적용 답변")
                rag_answer_output = gr.Textbox(
                    **create_textbox_config("지식베이스(DB) 참고 답변", lines=10, scrollable=True),
                    elem_classes=["scroll-area"]
                )
                
                # RAG 시간 표시
                rag_time_output = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_classes=["time-display"]
                )
                
                with gr.Accordion("🔍 상세 과정 보기", open=False):
                    rag_desc_output = gr.Textbox(
                        **create_textbox_config("이미지 분석 결과", lines=8, scrollable=True),
                        elem_classes=["scroll-area"]
                    )
                    rag_context_output = gr.Textbox(
                        **create_textbox_config("지식베이스 검색 결과", lines=8, scrollable=True),
                        elem_classes=["scroll-area"]
                    )

        # 성능 비교 섹션 추가
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 성능 분석")
                performance_output = gr.Markdown(
                    value="",
                    elem_classes=["performance-summary"]
                )

        # 이벤트 바인딩
        submit_button.click(
            fn=gradio_comparison_interface,
            inputs=[image_input, question_input],
            outputs=[
                vlm_answer_output, 
                vlm_reasoning_output, 
                rag_answer_output, 
                rag_desc_output, 
                rag_context_output,
                vlm_time_output,
                rag_time_output,
                performance_output
            ]
        )
        
        # 초기화 버튼 기능
        clear_button.click(
            fn=lambda: (None, "", "", "", "", "", "", "", ""),
            outputs=[
                image_input, 
                question_input, 
                vlm_answer_output, 
                vlm_reasoning_output, 
                rag_answer_output, 
                rag_desc_output, 
                rag_context_output,
                vlm_time_output,
                rag_time_output,
                performance_output
            ]
        )
        
        # 사용 가이드
        with gr.Accordion("📖 사용 가이드", open=False):
            gr.Markdown("""
            ### 사용 방법
            1. **이미지 업로드**: 분석하고자 하는 이미지를 업로드하세요
            2. **질문 입력**: 이미지에 대해 구체적이고 명확한 질문을 입력하세요
            3. **결과 비교**: 버튼을 클릭하여 VLM 단독 답변과 RAG 적용 답변을 비교하세요
            
            ### 팁
            - 구체적인 질문일수록 더 정확한 답변을 얻을 수 있습니다
            - 이미지의 해상도가 높을수록 더 나은 분석이 가능합니다
            - 각 답변의 근거와 과정도 함께 확인해보세요
            """)
    
    return demo

def main():
    """메인 함수"""
    # 모델 로딩
    success = model_manager.load_models()
    
    if not success:
        logger.error("모델 로딩에 실패했습니다. 애플리케이션을 종료합니다.")
        return
    
    # Gradio 앱 생성 및 실행
    demo = create_demo()
    
    try:
        demo.launch(
            server_name="0.0.0.0",  # 외부 접속 허용
            server_port=7860,       # 포트 명시
            share=False,            # 공개 링크 생성 안함
            debug=False,            # 프로덕션에서는 False
            show_error=True,        # 에러 표시
            quiet=False             # 로그 표시
        )
    except KeyboardInterrupt:
        logger.info("사용자에 의해 애플리케이션이 종료되었습니다.")
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()