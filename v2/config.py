
# v2/config.py
import argparse

# --- 모델 및 경로 설정 ---
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
VLM_MODEL_PATH = "AIDC-AI/Ovis2-8B"
LLM_MODEL_PATH = "Qwen/Qwen2-1.5B-Instruct"  # 키워드 추출용 LLM (신규)

PERSIST_DIRECTORY = '/home/aisw/Project/UST-ETRI-2025/chroma_db'
COLLECTION_NAMES = [
    "korean_knowledge_base_v3",
    "subway_line_info_v1" 
]

# --- 프롬프트 템플릿 ---

# 1. 이미지 설명을 생성하기 위한 프롬프트
IMAGE_DESCRIPTION_PROMPT = (
    "당신은 이미지 분석 전문가입니다. 주어진 이미지를 관찰하고, 이미지에 나타난 모든 텍스트, 표지판, 노선도, 색상, 상징물 등 "
    "대한민국 지하철 시스템과 관련된 모든 시각적 정보를 상세하고 구조적으로 설명해주세요. "
    "특히, 여러 정보가 복합적으로 나타나는 경우(예: 환승역 표지판) 관계를 명확히 하여 서술하세요. <image>"
)

# 2. 질문에서 키워드를 추출하기 위한 프롬프트
KEYWORD_EXTRACTION_PROMPT_TEMPLATE = """당신은 문장의 핵심을 파악하는 전문가입니다. 다음 [질문]에서 가장 중요한 명사 키워드들을 쉼표(,)로 구분하여 추출해주세요. 불필요한 조사나 서술어는 제외하고, 검색에 도움이 될 핵심 단어만 간결하게 나열하세요.

## 예시
질문: 이 역의 이름은 무엇이고, 어떤 노선이 다니나요?
추출: 역 이름, 노선

질문: 4호선에 스크린도어가 설치되어 있나요?
추출: 4호선, 스크린도어

## [질문]
{user_question}"""


# 3. 최종 답변 생성을 위한 Chain-of-Thought 프롬프트
FINAL_ANSWER_PROMPT_TEMPLATE = """당신은 대한민국 지하철 정보에 능통한 AI 어시스턴트입니다. 아래 [지침]과 주어진 정보를 종합하여 사용자의 질문에 답변하세요.

## [지침]
1. **단계적 추론**: 아래 **[추론 과정]**에 따라 단계별로 생각하고 답변을 구성하세요.
2. **정보 종합**: [이미지 설명]과 [참고 정보]를 모두 활용하여 가장 정확하고 완전한 답변을 만드세요.
3. **근거 제시**: 답변의 신뢰도를 높이기 위해, 어떤 정보(이미지, 참고 정보)를 근거로 사용했는지 명확히 언급하세요.
4. **추론 금지**: 제공된 정보 내에서만 답변해야 합니다. 정보가 부족하면 "제공된 정보만으로는 답변을 찾을 수 없습니다."라고 솔직하게 답변하세요.

---
## [제공된 정보]

**1. 이미지 설명**
{image_description}

**2. 참고 정보 (VectorDB 검색 결과)**
{context}

---
## [추론 과정]

**1. 질문의 핵심 의도 파악**:
   - 사용자의 질문: "{user_question}"
   - 이 질문의 핵심은 무엇인가? (예: 역 이름, 환승 정보, 위치 등)

**2. 정보 수집 및 분석**:
   - **이미지**: 이미지에서 질문과 관련된 시각적 단서(텍스트, 색상, 기호 등)를 찾는다.
   - **이미지 설명**: 생성된 이미지 설명에서 관련된 내용을 확인한다.
   - **참고 정보**: VectorDB에서 검색된 문서들 중 질문에 답할 수 있는 내용을 찾는다.

**3. 결론 도출**:
   - 위 정보들을 종합하여 질문에 대한 명확한 결론을 내린다.

**4. 답변 작성**:
   - 위 결론을 바탕으로, 아래 [포맷]에 맞춰 최종 답변과 근거를 작성한다.

---
## [포맷]

**[최종 답변]**
[여기에 최종 답변을 서술합니다.]

**[근거]**
- [사용한 근거 1]
- [사용한 근거 2]
...

---
<image>
"""


def get_args():
    """커맨드 라인 인자를 파싱하여 반환합니다."""
    parser = argparse.ArgumentParser(description="V2 - 하이브리드 RAG 파이프라인")
    parser.add_argument(
        "--image_path", 
        type=str, 
        default='/home/aisw/Project/UST-ETRI-2025/VLM_RAG/images/seoul_station.jpg', 
        help="이미지 파일 경로"
    )
    parser.add_argument(
        "--question", 
        type=str, 
        default='4호선 서울역의 스크린도어 설치 여부와 다음 역의 이름을 알려줘.',
        help="사용자 질문"
    )
    return parser.parse_args()
