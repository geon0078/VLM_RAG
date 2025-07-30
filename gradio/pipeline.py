# pipeline.py

import torch
import gradio as gr

def run_vlm_only_pipeline(
    image, 
    user_question, 
    vlm_model, 
    text_tokenizer,
    vis_tokenizer
):
    """RAG 없이 VLM 단독으로 이미지와 질문에 대해 답변합니다."""
    prompt_text = f"{user_question}\n<image>"
    
    prompt, input_ids, pixel_values = vlm_model.preprocess_inputs(prompt_text, [image], max_partition=9)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).unsqueeze(0).to(vlm_model.device)
    input_ids = input_ids.unsqueeze(0).to(vlm_model.device)
    pixel_values = [pixel_values.to(dtype=vis_tokenizer.dtype, device=vis_tokenizer.device)] if pixel_values is not None else None

    with torch.inference_mode():
        output_ids = vlm_model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, max_new_tokens=512, do_sample=False, eos_token_id=[text_tokenizer.eos_token_id, text_tokenizer.convert_tokens_to_ids("<|im_end|>")])[0]
        final_answer = text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    return final_answer


# pipeline.py 의 run_vlm_only_pipeline 함수 부분

def run_vlm_only_pipeline(
    image, 
    user_question, 
    vlm_model, 
    text_tokenizer,
    vis_tokenizer
):
    """
    RAG 없이 VLM 단독으로 답변하며, [근거]와 [최종 답변]을 분리하여 생성합니다.
    """
    # [수정] Chain-of-Thought 프롬프트 적용
    prompt_text = (
        "당신은 이미지 분석 전문가입니다. 주어진 [질문]과 [이미지]를 바탕으로, 아래 [포맷]에 맞춰 답변하세요.\n\n"
        "## [포맷]\n"
        "1. **[근거]**: 이미지를 관찰하고 질문에 답하기 위한 단계별 추론 과정을 설명합니다. 이미지에서 발견한 구체적인 시각적 단서를 바탕으로 서술하세요.\n"
        "2. **[최종 답변]**: 위 근거를 바탕으로 질문에 대한 명확하고 간결한 최종 결론을 내립니다.\n\n"
        "--- 아래 정보를 사용하세요 ---\n"
        f"[질문]: {user_question}\n"
        "<image>"
    )
    
    prompt, input_ids, pixel_values = vlm_model.preprocess_inputs(prompt_text, [image], max_partition=9)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).unsqueeze(0).to(vlm_model.device)
    input_ids = input_ids.unsqueeze(0).to(vlm_model.device)
    pixel_values = [pixel_values.to(dtype=vis_tokenizer.dtype, device=vis_tokenizer.device)] if pixel_values is not None else None

    with torch.inference_mode():
        output_ids = vlm_model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, max_new_tokens=1024, do_sample=False, eos_token_id=[text_tokenizer.eos_token_id, text_tokenizer.convert_tokens_to_ids("<|im_end|>")])[0]
        full_answer = text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
    # [수정] 모델의 답변에서 [근거]와 [최종 답변]을 파싱하여 분리
    try:
        reasoning = full_answer.split("[최종 답변]")[0].replace("[근거]", "").strip()
        final_answer = full_answer.split("[최종 답변]")[1].strip()
    except IndexError:
        # 모델이 포맷을 따르지 않은 경우, 전체 답변을 최종 답변으로 처리
        reasoning = "모델이 답변 포맷을 따르지 않아 근거를 분리할 수 없었습니다."
        final_answer = full_answer
    
    return final_answer, reasoning

def run_rag_pipeline(
    image, 
    user_question, 
    vlm_model, 
    text_tokenizer, 
    vis_tokenizer, 
    embedding_model, 
    chroma_collection,
    progress=gr.Progress(track_tqdm=True)
):
    """기존의 RAG 파이프라인을 실행합니다."""
    
    if chroma_collection is None:
        raise gr.Error("ChromaDB 컬렉션이 로드되지 않았습니다. 서버 로그를 확인해주세요.")


    progress(0, desc="[RAG] 이미지 설명 생성 중...")
    image_prompt = (
        "이 이미지는 대한민국 지하철과 관련된 사진입니다. "
        "노선, 역명, 환승, 색상, 주변 정보 등 지하철과 직접적으로 연관된 내용을 중심으로, "
        "이미지에서 파악할 수 있는 핵심 정보를 요약해 주세요. 불필요한 묘사나 일반적인 설명은 생략하고, "
        "지하철 시스템과 관련된 사실만 간결하게 정리해 주세요. <image>"
    )
    prompt, input_ids, pixel_values = vlm_model.preprocess_inputs(image_prompt, [image], max_partition=9)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).unsqueeze(0).to(vlm_model.device)
    input_ids = input_ids.unsqueeze(0).to(vlm_model.device)
    pixel_values_desc = [pixel_values.to(dtype=vis_tokenizer.dtype, device=vis_tokenizer.device)] if pixel_values is not None else None

    with torch.inference_mode():
        output_ids = vlm_model.generate(input_ids, pixel_values=pixel_values_desc, attention_mask=attention_mask, max_new_tokens=256, do_sample=False)[0]
        image_description = text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    progress(0.2, desc="[RAG] 이미지 설명 기반 벡터 검색 중...")
    image_query_embedding = embedding_model.encode([image_description]).tolist()
    image_results = chroma_collection.query(query_embeddings=image_query_embedding, n_results=3)
    image_retrieved_docs = image_results['documents'][0]
    image_context_str = "\n".join(image_retrieved_docs)

    progress(0.4, desc="[RAG] 사용자 질문 기반 벡터 검색 중...")
    question_query_embedding = embedding_model.encode([user_question]).tolist()
    question_results = chroma_collection.query(query_embeddings=question_query_embedding, n_results=3)
    question_retrieved_docs = question_results['documents'][0]
    question_context_str = "\n".join(question_retrieved_docs)

    # 두 검색 결과를 모두 참고정보로 넘김
    context_str = (
        "[이미지 설명 기반 검색 결과]\n" + (image_context_str if image_context_str.strip() else "없음") +
        "\n\n[질문 기반 검색 결과]\n" + (question_context_str if question_context_str.strip() else "없음")
    )

    progress(0.6, desc="[RAG] 최종 답변 생성 중...")
    final_prompt_text = (
        "당신은 유능한 철도 정보 분석가입니다. 아래 [지침]과 [참고 정보]를 활용하여 [문맥], [이미지] 정보로 [질문]에 답변하세요.\n\n"
        "## 지침\n"
        "1. **종합적 분석**: [문맥]에 여러 검색 결과가 있다면, 모든 내용을 종합적으로 분석하여 증거를 찾으세요.\n"
        "2. **환승역 처리**: 특정 역 이름이 여러 노선 정보와 함께 반복되면 '환승역'일 가능성이 높습니다. 이 경우, 관련된 모든 노선 정보를 종합하여 답변하세요.\n"
        "3. **엄격한 근거**: 반드시 [문맥], [이미지], [참고 정보]에서 찾은 명확한 근거를 바탕으로 답변해야 합니다. 추측하거나 없는 정보를 만들지 마세요.\n"
        "4. **정보 부재 시**: 분석 결과, 질문에 대한 답을 찾을 수 없다면, 반드시 \"지식베이스(VectorDB)와 이미지를 참고하였으나 답변을 찾을 수 없습니다.\"라고만 답변하세요.\n\n"
        "## 참고 정보: 서울 지하철 노선 색상 🎨\n"
        "- 1호선: 남색, 2호선: 초록색, 3호선: 주황색, 4호선: 하늘색, 5호선: 보라색, 6호선: 황토색, 7호선: 올리브색, 8호선: 분홍색, 9호선: 금색\n\n"
        "--- 아래 정보를 사용하세요 ---\n"
        f"[문맥]\n{context_str if context_str.strip() else '없음'}\n\n"
        f"[질문]\n{user_question}\n<image>"
    )

    prompt, input_ids, pixel_values_final = vlm_model.preprocess_inputs(final_prompt_text, [image], max_partition=9)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).unsqueeze(0).to(vlm_model.device)
    input_ids = input_ids.unsqueeze(0).to(vlm_model.device)
    pixel_values_final = [pixel_values_final.to(dtype=vis_tokenizer.dtype, device=vis_tokenizer.device)] if pixel_values_final is not None else None

    with torch.inference_mode():
        output_ids = vlm_model.generate(input_ids, pixel_values=pixel_values_final, attention_mask=attention_mask, max_new_tokens=512, do_sample=False, eos_token_id=[text_tokenizer.eos_token_id, text_tokenizer.convert_tokens_to_ids("<|im_end|>")])[0]
        final_answer = text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return final_answer, image_description, context_str