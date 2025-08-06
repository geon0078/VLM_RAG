# v3/module/retrieval.py
import torch
from ..config import IMAGE_DESCRIPTION_PROMPT, KEYWORD_EXTRACTION_PROMPT_TEMPLATE

def generate_image_description(image, vlm_model, vlm_text_tokenizer, vlm_vis_tokenizer):
    """VLM을 사용하여 이미지에 대한 설명을 생성합니다."""
    print("[Retrieval] 1. 이미지 설명 생성 중...")
    
    prompt, input_ids, pixel_values = vlm_model.preprocess_inputs(
        IMAGE_DESCRIPTION_PROMPT, [image], max_partition=9
    )
    attention_mask = torch.ne(input_ids, vlm_text_tokenizer.pad_token_id).unsqueeze(0).to(vlm_model.device)
    input_ids = input_ids.unsqueeze(0).to(vlm_model.device)
    pixel_values_desc = [pixel_values.to(dtype=vlm_vis_tokenizer.dtype, device=vlm_vis_tokenizer.device)] if pixel_values is not None else None

    with torch.inference_mode():
        output_ids = vlm_model.generate(
            input_ids, 
            pixel_values=pixel_values_desc, 
            attention_mask=attention_mask, 
            max_new_tokens=1024, 
            do_sample=False
        )[0]
        description = vlm_text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
    print(f"✅ 이미지 설명 생성 완료.")
    return description

def extract_keywords_from_question(user_question, llm_model, llm_tokenizer):
    """LLM을 사용하여 질문에서 핵심 키워드를 추출합니다."""
    print("[Retrieval] 2. 질문에서 키워드 추출 중 (LLM 사용)...")
    
    prompt_text = KEYWORD_EXTRACTION_PROMPT_TEMPLATE.format(user_question=user_question)
    
    messages = [{"role": "user", "content": prompt_text}]
    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

    with torch.inference_mode():
        generated_ids = llm_model.generate(model_inputs.input_ids, max_new_tokens=64)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        keywords_text = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    keywords = [kw.strip() for kw in keywords_text.split(',')]
    print(f"✅ 추출된 키워드: {keywords}")
    return keywords

def search_vectordb(queries, embedding_model, chroma_collections):
    """
    주어진 쿼리들로 ChromaDB에서 문서를 검색하고, 유사도 점수를 포함하여 반환합니다.
    """
    all_results = {}
    print(f"[Retrieval] 3. 확장된 쿼리로 벡터 검색 중... (총 {len(queries)}개 쿼리)")
    for query in queries:
        query_embedding = embedding_model.encode([query]).tolist()
        for cname, collection in chroma_collections.items():
            if collection is not None:
                try:
                    # include=['documents', 'distances']를 통해 점수(distance)를 함께 요청
                    results = collection.query(
                        query_embeddings=query_embedding, 
                        n_results=10, # 더 많은 후보군 확보
                        include=['documents', 'distances']
                    )
                    
                    # 결과의 각 문서에 대해 (문서 내용, 점수)를 저장
                    for doc_id, doc_content, distance in zip(results.get('ids', [[]])[0], results.get('documents', [[]])[0], results.get('distances', [[]])[0]):
                        # 점수는 distance이므로, 낮을수록 유사함 (0에 가까울수록 관련성 높음)
                        # 기존 결과가 없거나, 새 점수가 더 낮을 경우에만 업데이트
                        if doc_id not in all_results or distance < all_results[doc_id][1]:
                            all_results[doc_id] = (doc_content, distance)

                except Exception as e:
                    print(f"[Retrieval] 쿼리 '{query}'에 대한 컬렉션 '{cname}' 검색 오류: {e}")
    
    # 점수(distance)가 낮은 순서대로 정렬
    sorted_results = sorted(all_results.values(), key=lambda item: item[1])
    
    return sorted_results

def retrieve_context(
    image,
    user_question,
    models
):
    """
    이미지와 질문을 바탕으로 VectorDB에서 관련 문맥을 검색하고 중간 결과물을 반환합니다.
    """
    image_description = generate_image_description(
        image, models["vlm_model"], models["vlm_text_tokenizer"], models["vlm_vis_tokenizer"]
    )
    keywords = extract_keywords_from_question(
        user_question, models["llm_model"], models["llm_tokenizer"]
    )
    search_queries = [user_question, image_description] + keywords
    
    # 점수가 포함된 정렬된 결과 리스트를 받음
    scored_results = search_vectordb(search_queries, models["embedding_model"], models["collections"])
    
    if not scored_results:
        print("⚠️ [Retrieval] 벡터 검색 결과가 없습니다.")
        context_str = "없음"
    else:
        print(f"✅ 벡터 검색 완료. {len(scored_results)}개의 고유 문서 검색됨 (점수 기준 정렬).")
        # 컨텍스트 생성 시 점수를 포함
        context_lines = [f"(Score: {distance:.4f}) {doc}" for doc, distance in scored_results]
        context_str = "\n".join(context_lines)

    print("\n--- [V3] 검색된 문서 (점수 포함) ---")
    print(context_str)
    print("-------------------------------------\n")

    return {
        "image_description": image_description,
        "keywords": keywords,
        "search_queries": search_queries,
        "retrieved_context": context_str
    }
