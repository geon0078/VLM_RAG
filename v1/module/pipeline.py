# v1/module/pipeline.py
import torch

def run_rag_pipeline(
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

    queries = [user_question]
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
3. **추론 금지**: 제공된 정보 내에서만 답변해야 합니다. 정보가 부족하여 답변을 확신할 수 없는 경우, "제공된 정보만으로는 답변을 찾을 수 없습니다."라고 솔직하게 답변하세요.
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

    return final_answer, reasoning
