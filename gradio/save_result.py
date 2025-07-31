import os
import csv
import time
import logging
from PIL import Image
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from contextlib import contextmanager

# GPU 설정을 최상단으로 이동
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 두 번째 GPU만 사용

# 모델 및 파이프라인 불러오기
from models import load_all_models
from pipeline import run_vlm_only_pipeline, run_rag_pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 폴더 및 파일 설정
IMAGE_DIR = "./images"
QUESTION_CSV = "./questions.csv"
OUTPUT_CSV = "./vlm_rag_results.csv"

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
            logger.info("🚀 배치 처리 시작... 모든 모델 로딩을 시작합니다.")
            
            self.vlm, self.txt_tokenizer, self.vis_tokenizer, self.emb_model, self.collections = load_all_models()
            
            # 모델 검증
            if not self.validate_models():
                return False
            
            logger.info("✨ 모든 모델과 DB가 준비되었습니다. 배치 처리를 시작합니다.")
            self.models_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"🚨 치명적 오류: 모델 로딩에 실패했습니다. {e}")
            self.models_loaded = False
            return False
    
    def validate_models(self) -> bool:
        """모델들이 제대로 로드되었는지 확인"""
        checks = [
            (self.vlm, "VLM 모델"),
            (self.txt_tokenizer, "텍스트 토크나이저"),
            (self.vis_tokenizer, "비전 토크나이저"),
            (self.emb_model, "임베딩 모델"),
            (self.collections, "컬렉션")
        ]
        
        for model, name in checks:
            if model is None:
                logger.error(f"❌ {name}이 제대로 로드되지 않았습니다")
                return False
            logger.info(f"✅ {name} 검증 완료")
        
        return True
    
    def is_ready(self) -> bool:
        """모델이 준비되었는지 확인합니다."""
        return self.models_loaded

@contextmanager
def timer(operation_name: str):
    """시간 측정을 위한 컨텍스트 매니저"""
    start_time = time.time()
    try:
        yield start_time
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{operation_name} 소요 시간: {format_time(elapsed_time)}")

@contextmanager
def error_handler(operation_name: str):
    """에러 처리를 위한 컨텍스트 매니저"""
    try:
        yield
    except Exception as e:
        logger.error(f"{operation_name} 중 오류 발생: {e}")
        raise

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

def validate_inputs(image: Optional[Image.Image], question: str) -> None:
    """입력값 검증"""
    if image is None:
        raise ValueError("이미지를 로드할 수 없습니다.")
    
    if not question or question.strip() == "":
        raise ValueError("질문이 비어있습니다.")
    
    if len(question.strip()) < 3:
        raise ValueError("질문이 너무 짧습니다. 최소 3글자 이상 필요합니다.")

def load_questions_by_image(csv_path: str) -> Dict[str, List[str]]:
    """이미지별로 질문 리스트를 반환"""
    if not os.path.exists(csv_path):
        logger.error(f"❌ 질문 파일이 존재하지 않습니다: {csv_path}")
        return {}
    
    mapping = {}
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, 1):
                try:
                    img = row["image_name"].strip()
                    q = row["question"].strip()
                    if not img or not q:
                        logger.warning(f"빈 데이터 건너뛰기 (행 {row_num}): img='{img}', q='{q}'")
                        continue
                    mapping.setdefault(img, []).append(q)
                except KeyError as e:
                    logger.error(f"CSV 컬럼 오류 (행 {row_num}): {e}")
                    continue
    except Exception as e:
        logger.error(f"❌ CSV 파일 읽기 실패: {e}")
        return {}
    
    logger.info(f"📋 총 {len(mapping)}개 이미지에 대한 질문을 로드했습니다")
    return mapping

def safe_image_load(img_path: str) -> Optional[Image.Image]:
    """안전하게 이미지를 로드"""
    try:
        if not os.path.exists(img_path):
            logger.warning(f"⚠️ 이미지 파일 없음: {img_path}")
            return None
        
        image = Image.open(img_path).convert("RGB")
        logger.debug(f"✅ 이미지 로드 성공: {img_path} (크기: {image.size})")
        return image
        
    except Exception as e:
        logger.error(f"❌ 이미지 열기 실패: {img_path} - {e}")
        return None

def debug_model_state(model_manager: ModelManager, operation: str):
    """모델 상태를 디버깅하는 함수"""
    logger.info(f"🔍 {operation} - 모델 상태 체크:")
    logger.info(f"   - VLM: {type(model_manager.vlm)} ({'OK' if model_manager.vlm is not None else 'NONE'})")
    logger.info(f"   - txt_tokenizer: {type(model_manager.txt_tokenizer)} ({'OK' if model_manager.txt_tokenizer is not None else 'NONE'})")
    logger.info(f"   - vis_tokenizer: {type(model_manager.vis_tokenizer)} ({'OK' if model_manager.vis_tokenizer is not None else 'NONE'})")
    logger.info(f"   - emb_model: {type(model_manager.emb_model)} ({'OK' if model_manager.emb_model is not None else 'NONE'})")
    logger.info(f"   - collections: {type(model_manager.collections)} ({'OK' if model_manager.collections is not None else 'NONE'})")

def safe_run_vlm_pipeline(image, question, vlm, txt_tokenizer, vis_tokenizer):
    """VLM 파이프라인을 안전하게 실행"""
    try:
        # 파라미터 검증
        if vlm is None:
            raise ValueError("VLM 모델이 None입니다")
        if txt_tokenizer is None:
            raise ValueError("텍스트 토크나이저가 None입니다")
        if vis_tokenizer is None:
            raise ValueError("비전 토크나이저가 None입니다")
        
        logger.debug("VLM 파이프라인 파라미터 검증 완료")
        
        return run_vlm_only_pipeline(image, question, vlm, txt_tokenizer, vis_tokenizer)
    
    except Exception as e:
        logger.error(f"VLM 파이프라인 실행 중 오류: {e}")
        import traceback
        logger.error(f"상세 트레이스백:\n{traceback.format_exc()}")
        raise

class DummyProgress:
    """배치 처리용 더미 progress 객체"""
    def __call__(self, value, desc=""):
        logger.debug(f"[Progress] {desc} - {value*100:.1f}%")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def safe_run_rag_pipeline(image, question, vlm, txt_tokenizer, vis_tokenizer, emb_model, collections):
    """RAG 파이프라인을 안전하게 실행"""
    try:
        # 파라미터 검증
        if vlm is None:
            raise ValueError("VLM 모델이 None입니다")
        if txt_tokenizer is None:
            raise ValueError("텍스트 토크나이저가 None입니다")
        if vis_tokenizer is None:
            raise ValueError("비전 토크나이저가 None입니다")
        if emb_model is None:
            raise ValueError("임베딩 모델이 None입니다")
        if collections is None:
            raise ValueError("컬렉션이 None입니다")
        
        logger.debug("RAG 파이프라인 파라미터 검증 완료")
        
        # 컬렉션 상태 확인
        if hasattr(collections, '__len__'):
            logger.debug(f"컬렉션 개수: {len(collections)}")
        elif hasattr(collections, 'keys'):
            logger.debug(f"컬렉션 키: {list(collections.keys())}")
        
        # 더미 progress 객체 생성 (None 대신 사용)
        dummy_progress = DummyProgress()
        
        return run_rag_pipeline(
            image, question, vlm, txt_tokenizer, vis_tokenizer, 
            emb_model, collections, progress=dummy_progress
        )
    
    except Exception as e:
        logger.error(f"RAG 파이프라인 실행 중 오류: {e}")
        import traceback
        logger.error(f"상세 트레이스백:\n{traceback.format_exc()}")
        raise

def process_single_question(
    image: Image.Image, 
    question: str, 
    model_manager: ModelManager,
    img_name: str
) -> Dict[str, any]:
    """단일 질문을 처리하여 VLM과 RAG 결과를 모두 반환"""
    
    # 입력값 검증
    validate_inputs(image, question)
    question = question.strip()
    
    # 결과 저장용 딕셔너리
    result = {
        "image_name": img_name,
        "question": question,
        "vlm_answer": None,
        "vlm_reasoning": None,
        "vlm_time": 0,
        "rag_answer": None,
        "rag_reasoning": None,
        "rag_context": None,
        "rag_time": 0,
        "error": None
    }
    
    try:
        # 모델 상태 디버깅
        debug_model_state(model_manager, "질문 처리 시작")
        
        # VLM 단독 답변 생성 (시간 측정)
        logger.info(f"🧠 VLM 처리 시작: '{question[:50]}...'")
        
        vlm_start_time = time.time()
        try:
            vlm_answer, vlm_reasoning = safe_run_vlm_pipeline(
                image, 
                question, 
                model_manager.vlm, 
                model_manager.txt_tokenizer, 
                model_manager.vis_tokenizer
            )
            vlm_end_time = time.time()
            vlm_time = vlm_end_time - vlm_start_time
            
            result.update({
                "vlm_answer": vlm_answer,
                "vlm_reasoning": vlm_reasoning,
                "vlm_time": vlm_time
            })
            
            logger.info(f"VLM 처리 완료: {format_time(vlm_time)}")
            
        except Exception as e:
            logger.error(f"VLM 파이프라인 실패: {e}")
            result.update({
                "vlm_answer": f"VLM 처리 실패: {str(e)}",
                "vlm_reasoning": "VLM 파이프라인에서 오류 발생",
                "vlm_time": 0
            })
        
        # RAG 적용 파이프라인 실행 (시간 측정)
        logger.info(f"📚 RAG 처리 시작: '{question[:50]}...'")
        
        rag_start_time = time.time()
        try:
            rag_answer, rag_desc, rag_context = safe_run_rag_pipeline(
                image, 
                question, 
                model_manager.vlm, 
                model_manager.txt_tokenizer, 
                model_manager.vis_tokenizer, 
                model_manager.emb_model, 
                model_manager.collections
            )
            rag_end_time = time.time()
            rag_time = rag_end_time - rag_start_time
            
            result.update({
                "rag_answer": rag_answer,
                "rag_reasoning": rag_desc,
                "rag_context": rag_context,
                "rag_time": rag_time
            })
            
            logger.info(f"RAG 처리 완료: {format_time(rag_time)}")
            
        except Exception as e:
            logger.error(f"RAG 파이프라인 실패: {e}")
            result.update({
                "rag_answer": f"RAG 처리 실패: {str(e)}",
                "rag_reasoning": "RAG 파이프라인에서 오류 발생",
                "rag_context": f"오류: {str(e)}",
                "rag_time": 0
            })
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ 질문 처리 실패: {error_msg}")
        import traceback
        logger.error(f"전체 트레이스백:\n{traceback.format_exc()}")
        
        result["error"] = error_msg
        
        # 실패한 경우 기본값 설정
        for key in ["vlm_answer", "vlm_reasoning", "rag_answer", "rag_reasoning", "rag_context"]:
            if result[key] is None:
                result[key] = f"처리 실패: {error_msg}"
    
    return result

def create_performance_summary(results: List[Dict]) -> str:
    """전체 처리 결과에 대한 성능 요약을 생성"""
    if not results:
        return "처리된 결과가 없습니다."
    
    # 성공한 결과만 필터링
    successful_results = [r for r in results if r["error"] is None]
    
    if not successful_results:
        return "성공적으로 처리된 결과가 없습니다."
    
    total_vlm_time = sum(r["vlm_time"] for r in successful_results)
    total_rag_time = sum(r["rag_time"] for r in successful_results)
    total_time = total_vlm_time + total_rag_time
    
    avg_vlm_time = total_vlm_time / len(successful_results)
    avg_rag_time = total_rag_time / len(successful_results)
    
    success_rate = len(successful_results) / len(results) * 100
    
    summary = f"""
📊 **배치 처리 성능 요약**

📈 **처리 통계**
• 총 질문 수: {len(results)}개
• 성공 처리: {len(successful_results)}개 ({success_rate:.1f}%)
• 실패 처리: {len(results) - len(successful_results)}개

⏱️ **시간 분석**
• VLM 총 시간: {format_time(total_vlm_time)}
• RAG 총 시간: {format_time(total_rag_time)}
• 전체 처리 시간: {format_time(total_time)}

⚡ **평균 처리 시간**
• VLM 평균: {format_time(avg_vlm_time)}
• RAG 평균: {format_time(avg_rag_time)}
"""
    
    if avg_vlm_time > 0 and avg_rag_time > 0:
        speedup = avg_vlm_time / avg_rag_time
        if speedup > 1:
            summary += f"\n🚀 **성능 비교**: VLM이 RAG보다 평균 {speedup:.1f}배 빠름"
        else:
            summary += f"\n🚀 **성능 비교**: RAG가 VLM보다 평균 {1/speedup:.1f}배 빠름"
    
    return summary

def main():
    """메인 함수"""
    logger.info("🚀 배치 이미지 처리 스크립트 시작")
    
    # 출력 디렉토리 확인
    os.makedirs(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else ".", exist_ok=True)
    
    # 모델 매니저 초기화 및 로딩
    model_manager = ModelManager()
    
    if not model_manager.load_models():
        logger.error("❌ 모델 로딩에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    # 질문 데이터 로드
    qa_map = load_questions_by_image(QUESTION_CSV)
    if not qa_map:
        logger.error("❌ 처리할 질문이 없습니다.")
        return

    results = []
    total_questions = sum(len(questions) for questions in qa_map.values())
    logger.info(f"📊 총 {total_questions}개의 질문을 처리합니다")

    processed_count = 0
    error_count = 0

    with timer("전체 배치 처리"):
        for img_name, questions in tqdm(qa_map.items(), desc="🔍 이미지 처리 중"):
            img_path = os.path.join(IMAGE_DIR, img_name)
            
            # 이미지 로드
            image = safe_image_load(img_path)
            if image is None:
                # 이미지 로드 실패시 모든 질문에 대해 오류 기록
                for question in questions:
                    results.append({
                        "image_name": img_name,
                        "question": question,
                        "vlm_answer": "이미지 로드 실패",
                        "vlm_reasoning": "이미지를 불러올 수 없습니다",
                        "rag_answer": "이미지 로드 실패",
                        "rag_reasoning": "이미지를 불러올 수 없습니다",
                        "rag_context": "없음"
                    })
                    error_count += 1
                continue

            # 각 질문 처리
            for question in questions:
                processed_count += 1
                logger.info(f"🔄 처리 중 [{processed_count}/{total_questions}]: {img_name} - '{question[:50]}...'")
                
                # 단일 질문 처리
                result = process_single_question(image, question, model_manager, img_name)
                
                # CSV 출력용 형태로 변환
                csv_result = {
                    "image_name": result["image_name"],
                    "question": result["question"],
                    "vlm_answer": result["vlm_answer"] or "처리 실패",
                    "vlm_reasoning": result["vlm_reasoning"] or "추론 없음",
                    "rag_answer": result["rag_answer"] or "처리 실패",
                    "rag_reasoning": result["rag_reasoning"] or "추론 없음",
                    "rag_context": result["rag_context"] or "컨텍스트 없음"
                }
                
                results.append(csv_result)
                
                if result["error"] is not None:
                    error_count += 1

    # 결과 CSV 저장
    fieldnames = [
        "image_name", "question",
        "vlm_answer", "vlm_reasoning",
        "rag_answer", "rag_reasoning", "rag_context"
    ]

    try:
        with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"✅ 모든 처리 완료!")
        logger.info(f"📁 결과 저장 위치: {OUTPUT_CSV}")
        logger.info(f"📊 처리 통계:")
        logger.info(f"   - 총 처리된 질문: {processed_count}")
        logger.info(f"   - 성공: {processed_count - error_count}")
        logger.info(f"   - 오류: {error_count}")
        
        # 성능 요약 출력 (로그용)
        if results:
            # 시간 정보가 있는 결과들로 성능 요약 생성 (실제로는 모든 결과에 시간 정보가 없으므로 기본 통계만)
            success_rate = (processed_count - error_count) / processed_count * 100 if processed_count > 0 else 0
            logger.info(f"🎯 최종 성공률: {success_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ 결과 저장 실패: {e}")

if __name__ == "__main__":
    main()