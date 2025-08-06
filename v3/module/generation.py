
# v2/module/generation.py
import torch
from config import FINAL_ANSWER_PROMPT_TEMPLATE

def generate_final_answer(
    image,
    user_question,
    image_description,
    context,
    vlm_model,
    text_tokenizer,
    vis_tokenizer
):
    """
    모든 정보를 종합하고 CoT 프롬프트를 사용하여 최종 답변을 생성합니다.
    """
    print("[Generation] 3. 최종 답변 생성 중...")

    # CoT 프롬프트 템플릿에 정보 채우기
    final_prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(
        user_question=user_question,
        image_description=image_description,
        context=context
    )

    # VLM으로 최종 답변 생성
    prompt, input_ids, pixel_values = vlm_model.preprocess_inputs(
        final_prompt, [image], max_partition=9
    )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).unsqueeze(0).to(vlm_model.device)
    input_ids = input_ids.unsqueeze(0).to(vlm_model.device)
    pixel_values_final = [pixel_values.to(dtype=vis_tokenizer.dtype, device=vis_tokenizer.device)] if pixel_values is not None else None

    with torch.inference_mode():
        output_ids = vlm_model.generate(
            input_ids,
            pixel_values=pixel_values_final,
            attention_mask=attention_mask,
            max_new_tokens=2048,  # 더 길고 상세한 답변을 위해 토큰 수 증가
            do_sample=False,
            eos_token_id=[text_tokenizer.eos_token_id, text_tokenizer.convert_tokens_to_ids("<|im_end|>")]
        )[0]
        final_answer_full = text_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    print("✅ 최종 답변 생성 완료.")
    
    # 답변과 근거 분리
    try:
        # '[최종 답변]'과 '[근거]' 사이의 내용을 최종 답변으로 추출
        answer_part = final_answer_full.split("**[최종 답변]**")[1]
        final_answer = answer_part.split("**[근거]**")[0].strip()
        
        # '[근거]' 이후의 내용을 근거로 추출
        reasoning = final_answer_full.split("**[근거]**")[1].strip()

    except IndexError:
        # 모델이 포맷을 따르지 않은 경우, 전체를 최종 답변으로 처리
        final_answer = final_answer_full
        reasoning = "모델이 답변 포맷을 따르지 않아 근거를 분리할 수 없었습니다."

    return final_answer, reasoning
