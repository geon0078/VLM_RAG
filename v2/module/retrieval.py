# v2/module/retrieval.py
import torch
from config import IMAGE_DESCRIPTION_PROMPT, KEYWORD_EXTRACTION_PROMPT_TEMPLATE

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
    """주어진 쿼리들로 ChromaDB에서 문서를 검색합니다."""
    all_retrieved_docs = {}
    print(f"[Retrieval] 3. 확장된 쿼리로 벡터 검색 중... (총 {len(queries)}개 쿼리)")
    for query in queries:
        query_embedding = embedding_model.encode([query]).tolist()
        for cname, collection in chroma_collections.items():
            if collection is not None:
                try:
                    results = collection.query(query_embeddings=query_embedding, n_results=5)
                    for doc_id, doc_content in zip(results.get('ids', [[]])[0], results.get('documents', [[]])[0]):
                        all_retrieved_docs[doc_id] = doc_content
                except Exception as e:
                    print(f"[Retrieval] 쿼리 '{query}'에 대한 컬렉션 '{cname}' 검색 오류: {e}")
    
    return list(all_retrieved_docs.values())

def retrieve_context(
    image,
    user_question,
    models
):
    """
    이미지와 질문을 바탕으로 VectorDB에서 관련 문맥을 검색하고 중간 결과물을 반환합니다.
    """
    # 1. VLM으로 이미지 설명 생성
    image_description = generate_image_description(
        image, models["vlm_model"], models["vlm_text_tokenizer"], models["vlm_vis_tokenizer"]
    )

    # 2. LLM으로 질문에서 키워드 추출
    keywords = extract_keywords_from_question(
        user_question, models["llm_model"], models["llm_tokenizer"]
    )

    # 3. 검색 쿼리 확장: [원본 질문, 이미지 설명, 키워드1, 키워드2, ...]
    search_queries = [user_question, image_description] + keywords
    
    retrieved_docs = search_vectordb(search_queries, models["embedding_model"], models["collections"])
    
    if not retrieved_docs:
        print("⚠️ [Retrieval] 벡터 검색 결과가 없습니다.")
        context_str = "없음"
    else:
        print(f"✅ 벡터 검색 완료. {len(retrieved_docs)}개의 고유 문서 검색됨.")
        context_str = "\n".join(retrieved_docs)

    # 모든 중간 결과물을 딕셔너리로 반환
    return {
        "image_description": image_description,
        "keywords": keywords,
        "search_queries": search_queries,
        "retrieved_context": context_str
    }