import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
import argparse
from PIL import Image

# --- config.py 내용 ---
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
VLM_MODEL_PATH = "AIDC-AI/Ovis2-8B"
PERSIST_DIRECTORY = '/home/aisw/Project/UST-ETRI-2025/chroma_db'
COLLECTION_NAMES = [
    "korean_knowledge_base_v2",
    "subway_multilang_all_lines"
]

# --- models.py 내용 ---
def load_all_models():
    """
    애플리케이션에 필요한 모든 모델과 DB 커넥션을 로드합니다.
    """
    print("VLM 모델을 로드하는 중...")
    vlm_model = AutoModelForCausalLM.from_pretrained(
        VLM_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    text_tokenizer = vlm_model.get_text_tokenizer()
    vis_tokenizer = vlm_model.get_visual_tokenizer()
    print("✅ VLM 모델 로드 완료.")

    print("임베딩 모델과 ChromaDB를 로드하는 중...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    collections = {}
    for cname in COLLECTION_NAMES:
        try:
            cobj = client.get_collection(name=cname)
            print(f"✅ ChromaDB 컬렉션 '{cname}' 로드 완료 ({cobj.count()}개 문서).")
            collections[cname] = cobj
        except Exception as e:
            print(f"🚨 '{cname}' 컬렉션 로드 실패: {e}")
            collections[cname] = None
    print("✅ 임베딩 모델 및 ChromaDB 로드 완료.")

    return vlm_model, text_tokenizer, vis_tokenizer, embedding_model, collections

# --- pipeline.py 내용 ---
def run_rag_pipeline_v2(
    image, 
    user_question, 
    vlm_model, 
    text_tokenizer, 
    vis_tokenizer, 
    embedding_model, 
    chroma_collections
):
    """
    개선된 RAG 파이프라인 (RAG-Fusion 스타일)
    """
    # if not chroma_collections or not isinstance(chroma_collections, dict) or all(v is None for v in chroma_collections.values()):
    #     raise ValueError("ChromaDB 컬렉션이 로드되지 않았습니다.")

    # 1. 이미지에서 텍스트 설명 생성 (CoT 프롬프트 사용)
    print("[RAG] 이미지 설명 생성 중...")
    image_prompt = (
        "당신은 이미지 분석 전문가입니다. 주어진 이미지를 관찰하고, 이미지에 나타난 모든 텍스트, 표지판, 노선도, 색상, 상징물 등 "
        "대한민국 지하철 시스템과 관련된 모든 시각적 정보를 상세하고 구조적으로 설명해주세요. "
        "특히, 여러 정보가 복합적으로 나타나는 경우(예: 환승역 표지판) 관계를 명확히 하여 서술하세요. <image>"
    )
    prompt, input_ids, pixel_values = vlm_model.preprocess_inputs(image_prompt, [image], max_partition=9)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).unsqueeze(0).to(vlm_model.device)
    input_ids = input_ids.unsqueeze(0).to(vlm_model.device)
    pixel_values_desc = [pixel_values.to(dtype=vis_tokenizer.dtype, device=vis_tokenizer.device)] if pixel_values is not None else None

    with torch.inference_mode():
        output_ids = vlm_model.generate(input_ids, pixel_values=pixel_values_desc, attention_mask=attention_mask, max_new_tokens=1024, do_sample=False)[0]
        image_description = text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print(f"✅ 이미지 설명 생성 완료:\n{image_description}")

    # 2. 사용자 질문과 이미지 설명을 기반으로 여러 개의 검색 쿼리 생성 (비활성화)
    # print("[RAG] 검색 쿼리 생성 중...")
    # query_generation_prompt = (
    #     f"당신은 똑똑한 검색 쿼리 생성기입니다. 사용자의 원본 질문과 이미지 설명을 바탕으로, VectorDB에서 관련 정보를 효과적으로 찾을 수 있는 다양한 검색 쿼리 3개를 생성해주세요.\n\n"
    #     f"**[원본 질문]**: {user_question}\n"
    #     f"**[이미지 설명]**: {image_description}\n\n"
    #     f"**[지침]**:\n"
    #     f"- 원본 질문의 핵심 의도를 유지하면서, 다른 관점에서 질문을 재구성하세요.\n"
    #     f"- 이미지 설명에서 중요한 키워드(역 이름, 노선 번호, 방향 등)를 활용하여 구체적인 쿼리를 만드세요.\n"
    #     f"- 생성된 쿼리들은 서로 다른 형태와 키워드를 포함하도록 다양화하세요.\n"
    #     f"- 각 쿼리는 한 줄로 작성하고, 구분자 없이 내용만 제시하세요."
    # )
    # 
    # prompt, input_ids, _ = vlm_model.preprocess_inputs(query_generation_prompt, [], max_partition=1)
    # attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).unsqueeze(0).to(vlm_model.device)
    # input_ids = input_ids.unsqueeze(0).to(vlm_model.device)

    # with torch.inference_mode():
    #     output_ids = vlm_model.generate(input_ids, pixel_values=[], attention_mask=attention_mask, max_new_tokens=256, do_sample=False)[0]
    #     generated_queries_text = text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    queries = [user_question] #+ [q.strip() for q in generated_queries_text.split('\n') if q.strip()]
    print(f"✅ 생성된 검색 쿼리:\n{queries}")

    # 3. 생성된 모든 쿼리로 VectorDB 검색 수행
    print(f"[RAG] {len(queries)}개 쿼리로 벡터 검색 중...")
    all_retrieved_docs = {}
    for query in queries:
        query_embedding = embedding_model.encode([query]).tolist()
        for cname, collection in chroma_collections.items():
            if collection is not None:
                try:
                    results = collection.query(query_embeddings=query_embedding, n_results=3)
                    for doc_id, doc_content in zip(results.get('ids', [[]])[0], results.get('documents', [[]])[0]):
                        all_retrieved_docs[doc_id] = doc_content
                except Exception as e:
                    print(f"[RAG] 쿼리 '{query}'에 대한 컬렉션 '{cname}' 검색 오류: {e}")

    context_str = "\n".join(all_retrieved_docs.values())
    print(f"✅ 벡터 검색 완료. {len(all_retrieved_docs)}개의 고유 문서 검색됨.")

    # 5. 최종 답변 생성
    print("[RAG] 최종 답변 생성 중...")
    final_prompt_text = f"""당신은 대한민국 지하철 정보에 능통한 AI 어시스턴트입니다. 아래 [지침]에 따라, 주어진 [참고 정보], [이미지 설명], [이미지]를 종합하여 사용자의 [질문]에 답변하세요.

## [지침]
1. **정보 종합**: [참고 정보]는 여러 검색 쿼리를 통해 수집된 다양한 문서들의 집합입니다. 모든 문서를 종합적으로 검토하여 질문에 대한 가장 정확하고 완전한 답변을 구성하세요.
2. **근거 제시**: 답변의 신뢰도를 높이기 위해, 어떤 [참고 정보]나 [이미지]의 시각적 단서를 근거로 사용했는지 명확히 언급하세요. (예: '충무로역 정보에 따르면...', '이미지 속 4호선 안내 표지판을 보면...')
3. **추론 금지**: 제공된 정보 내에서만 답변해야 합니다. 정보가 부족하여 답변을 확신할 수 없는 경우, \"제공된 정보만으로는 답변을 찾을 수 없습니다.\"라고 솔직하게 답변하세요.
4. **답변 형식**: 최종 답변은 아래 [포맷]을 반드시 준수하여 작성하세요.

## [포맷]
**[최종 답변]**
[여기에 질문에 대한 최종 답변을 서술합니다.]

**[근거]**
- [사용한 근거 1]
- [사용한 근거 2]
...

--- 아래 정보를 사용하세요 ---
**[질문]**: {user_question}

**[이미지 설명]**:
{image_description}

**[참고 정보]**:
{context_str if context_str.strip() else '없음'}

<image>"""

    prompt, input_ids, pixel_values_final = vlm_model.preprocess_inputs(final_prompt_text, [image], max_partition=9)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).unsqueeze(0).to(vlm_model.device)
    input_ids = input_ids.unsqueeze(0).to(vlm_model.device)
    pixel_values_final = [pixel_values_final.to(dtype=vis_tokenizer.dtype, device=vis_tokenizer.device)] if pixel_values_final is not None else None

    with torch.inference_mode():
        output_ids = vlm_model.generate(input_ids, pixel_values=pixel_values_final, attention_mask=attention_mask, max_new_tokens=1024, do_sample=False, eos_token_id=[text_tokenizer.eos_token_id, text_tokenizer.convert_tokens_to_ids("<|im_end|>")])[0]
        final_answer_full = text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # 답변과 근거 분리
    try:
        final_answer = final_answer_full.split("**[근거]**")[0].replace("**[최종 답변]**", "").strip()
        reasoning = final_answer_full.split("**[근거]**")[1].strip()
    except IndexError:
        final_answer = final_answer_full
        reasoning = "근거를 분리할 수 없었습니다."

    return final_answer, reasoning, image_description, context_str

def main():
    parser = argparse.ArgumentParser(description="VLM RAG 파이프라인")
    parser.add_argument("--image_path", type=str, required=True, help="이미지 파일 경로")
    parser.add_argument("--question", type=str, required=True, help="사용자 질문")
    args = parser.parse_args()

    # 모델 로드
    vlm_model, text_tokenizer, vis_tokenizer, embedding_model, collections = load_all_models()

    # 이미지 로드
    try:
        image = Image.open(args.image_path).convert('RGB')
    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {args.image_path}")
        return

    # RAG 파이프라인 실행
    final_answer, reasoning, _, _ = run_rag_pipeline_v2(
        image,
        args.question,
        vlm_model,
        text_tokenizer,
        vis_tokenizer,
        embedding_model,
        collections
    )

    # 결과 출력
    print("\n" + "="*50)
    print("                최종 결과")
    print("="*50)
    print(f"\n[질문] {args.question}\n")
    print(f"[최종 답변]\n{final_answer}\n")
    print(f"[근거]\n{reasoning}")
    print("="*50)


if __name__ == "__main__":
    main()
