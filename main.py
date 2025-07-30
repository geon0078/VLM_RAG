import os
from PIL import Image
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM
import torch
import chromadb # [추가] chromadb 라이브러리 직접 임포트
from sentence_transformers import SentenceTransformer # [추가] sentence-transformers 임포트

# 1. 임베딩 모델 준비 (SentenceTransformer 직접 사용)
embedding_model_name = 'BAAI/bge-m3'
embedding_model = SentenceTransformer(embedding_model_name)

# 2. ChromaDB 벡터스토어 직접 연결
persist_dir = './chroma_db'
client = chromadb.PersistentClient(path=persist_dir)
# 'my_collection'은 실제 컬렉션 이름으로 변경해야 할 수 있습니다.
collection = client.get_or_create_collection(name="korean_knowledge_base") 

# 3. Ovis VLM 모델 및 토크나이저 로드
model_path = "AIDC-AI/Ovis2-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    trust_remote_code=True,
    cache_dir="./hf_cache",
    device_map="auto",
    low_cpu_mem_usage=True,
)
tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# 4. [추가] 사용자 정의 문서 검색 함수
def retrieve_documents(query, k=3):
    """지정된 쿼리를 사용하여 ChromaDB에서 문서를 검색합니다."""
    # 쿼리를 임베딩
    query_embedding = embedding_model.encode([query]).tolist()
    
    # 컬렉션에서 쿼리 실행
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    
    # 검색된 문서를 반환
    return results['documents'][0]

# 5. 이미지→설명→문맥검색→답변 함수
def image_rag_manual(image_path, user_question):
    # (1) 이미지 로드 및 설명 생성
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}.")
        images = [Image.new('RGB', (512, 512), color='blue')]
    else:
        images = [Image.open(image_path)]
    image_only_prompt = "이 이미지를 보고 간단히 설명해 주세요. <image>"
    max_partition = 9
    
    prompt, input_ids, pixel_values = model.preprocess_inputs(image_only_prompt, images, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id).unsqueeze(0).to(model.device)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    if pixel_values is not None:
        pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

    with torch.inference_mode():
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, max_new_tokens=256, do_sample=False)[0]
        image_description = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print("--- 이미지 설명 ---")
    print(image_description)

    # (2) [수정] 사용자 정의 함수로 벡터DB 검색
    retrieved_docs = retrieve_documents(image_description) # langchain retriever 대신 직접 호출
    context_str = "\n".join(retrieved_docs) # .page_content가 필요 없음
    print("\n--- 검색된 문맥 (VectorDB) ---")
    print(context_str)

    # (3) 최종 답변 생성을 위한 프롬프트 조합
    prompt_template = PromptTemplate.from_template(
        "아래 [문맥]과 [이미지]에 주어진 정보만을 근거로 [질문]에 답변하세요.\n"
        "- 반드시 [문맥]의 내용을 직접 인용하거나 요약하여 답변하세요.\n"
        "- [문맥]에 답이 없거나 불충분하면, [이미지]를 참고하되 모르면 반드시 아래 예시처럼만 답변하세요:\n"
        "  'VectorDB(지식베이스)와 이미지를 참고하였으나 답변을 찾을 수 없습니다.'\n"
        "- [문맥]이나 [이미지]에 없는 내용을 상상하거나 지어내지 마세요. 추가 설명도 하지 마세요.\n\n"
        "[문맥]\n{context}\n\n[질문]\n{question}\n<image>"
    )
    final_prompt_text = prompt_template.format(context=context_str if context_str.strip() else '없음', question=user_question)

    # 최종 답변 생성을 위한 전처리
    prompt, input_ids, pixel_values_final = model.preprocess_inputs(final_prompt_text, images=images, max_partition=max_partition)
    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id).unsqueeze(0).to(model.device)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    
    if pixel_values_final is not None:
        pixel_values_final = [pixel_values_final.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
    else:
        pixel_values_final = [None] 

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            pixel_values=pixel_values_final, 
            attention_mask=attention_mask, 
            max_new_tokens=512, 
            do_sample=False, 
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
        )[0]
        final_answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    print("\n--- 최종 답변 ---")
    print(final_answer)
    return final_answer

if __name__ == "__main__":
    image_path = './data/Daejeon_station.jpeg'
    user_question = "대전역의 역사 전화번호는 어떻게 되나요?"
    image_rag_manual(image_path, user_question)