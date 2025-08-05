
import os
from PIL import Image
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM
import torch
import chromadb
from sentence_transformers import SentenceTransformer

class RagPipeline:
    def __init__(self, embedding_model_name='BAAI/bge-m3', vlm_model_path='AIDC-AI/Ovis2-8B', db_path='./chroma_db', collection_name='korean_knowledge_base'):
        # 1. 임베딩 모델 준비
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # 2. ChromaDB 벡터스토어 연결
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_or_create_collection(name=collection_name)

        # 3. Ovis VLM 모델 및 토크나이저 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            vlm_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            cache_dir="./hf_cache",
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def retrieve_documents(self, query, k=3):
        """지정된 쿼리를 사용하여 ChromaDB에서 문서를 검색합니다."""
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        return results['documents'][0]

    def process(self, image_path, user_question):
        # (1) 이미지 로드 및 설명 생성
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}.")
            images = [Image.new('RGB', (512, 512), color='blue')]
        else:
            images = [Image.open(image_path)]
        image_only_prompt = "이 이미지를 보고 간단히 설명해 주세요. <image>"
        max_partition = 9
        
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(image_only_prompt, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id).unsqueeze(0).to(self.model.device)
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        if pixel_values is not None:
            pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, max_new_tokens=256, do_sample=False)[0]
            image_description = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print("--- 이미지 설명 ---")
        print(image_description)

        # (2) 벡터DB 검색
        retrieved_docs = self.retrieve_documents(image_description)
        context_str = "\n".join(retrieved_docs)
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
        prompt, input_ids, pixel_values_final = self.model.preprocess_inputs(final_prompt_text, images=images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id).unsqueeze(0).to(self.model.device)
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        
        if pixel_values_final is not None:
            pixel_values_final = [pixel_values_final.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]
        else:
            pixel_values_final = [None]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values_final,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
            )[0]
            final_answer = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        print("\n--- 최종 답변 ---")
        print(final_answer)
        return final_answer
