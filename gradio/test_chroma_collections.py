# test_chroma_collections.py

import gradio as gr
from models import load_all_models

# 모델 및 컬렉션 로드
VLM, TXT_TOKENIZER, VIS_TOKENIZER, EMB_MODEL, COLLECTIONS = load_all_models()

# 테스트용 함수: 각 컬렉션에서 임의 쿼리로 top-3 결과 확인
def test_collection_search(query: str):
    results = {}
    for cname, collection in COLLECTIONS.items():
        if collection is not None:
            try:
                q_emb = EMB_MODEL.encode([query]).tolist()
                res = collection.query(query_embeddings=q_emb, n_results=3)
                docs = res.get('documents', [[]])[0]
                results[cname] = docs
            except Exception as e:
                results[cname] = f"검색 오류: {e}"
        else:
            results[cname] = "컬렉션 로드 실패 또는 없음"
    return results

iface = gr.Interface(
    fn=test_collection_search,
    inputs=gr.Textbox(label="테스트 쿼리 입력", value="서울역"),
    outputs=gr.JSON(label="컬렉션별 top-3 검색 결과"),
    title="ChromaDB 컬렉션별 검색 테스트",
    description="각 컬렉션에서 쿼리로 top-3 결과를 확인합니다."
)

if __name__ == "__main__":
    iface.launch()
