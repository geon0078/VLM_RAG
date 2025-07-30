import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import chromadb
import os

# --- 설정부 (Configuration) ---
# 데이터 소스가 되는 엑셀 파일 경로
EXCEL_FILE_PATH = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG/data/korean_train_20250101.xlsx'
# 사용할 임베딩 모델 이름
EMBEDDING_MODEL = 'BAAI/bge-m3'
# ChromaDB를 저장할 디렉터리 경로
PERSIST_DIRECTORY = './chroma_db'
# 저장할 컬렉션 이름 (main.py와 동일해야 함)
COLLECTION_NAME = "korean_knowledge_base_v2"

def main():
    """
    Excel 파일로부터 데이터를 읽어 ChromaDB 벡터 데이터베이스를 생성하고 저장합니다.
    """
    # 1. 데이터 로드 및 전처리
    print(f"📄 1단계: '{EXCEL_FILE_PATH}' 엑셀 파일 읽기 시작...")
    try:
        df = pd.read_excel(EXCEL_FILE_PATH)
        print("✅ 엑셀 파일 읽기 완료.")

        print("🔄 2단계: 텍스트 데이터 결합 중...")
        # 모든 컬럼을 문자열로 합쳐 '임베딩텍스트'라는 새 컬럼 생성
        # all_columns = df.columns.tolist()
        # df['임베딩텍스트'] = df[all_columns].astype(str).agg(' / '.join, axis=1)
        all_columns = df.columns.tolist()
        df['임베딩텍스트'] = df.apply(
            lambda row: ' / '.join([f"{col}:{row[col]}" for col in all_columns]),
            axis=1)
        documents = df['임베딩텍스트'].tolist()
        print(f"✅ 총 {len(documents)}개의 문서를 준비했습니다.")

    except FileNotFoundError:
        print(f"⚠️ 경고: '{EXCEL_FILE_PATH}' 파일을 찾을 수 없습니다. 예시 데이터로 대체합니다.")
        documents = [
            "광장시장은 대한민국 서울특별시 종로구에 위치한 전통 시장이다.",
            "1905년에 개설되었으며, 대한민국 최초의 상설 시장으로 알려져 있다.",
            "주요 판매 품목은 한복, 직물, 구제 의류, 그리고 다양한 먹거리이다.",
            "특히 빈대떡, 마약김밥, 육회 등이 유명하여 많은 관광객들이 찾는다."
        ]
    except Exception as e:
        print(f"🚨 오류 발생: {e}. 예시 데이터로 대체합니다.")
        documents = [
            "광장시장은 대한민국 서울특별시 종로구에 위치한 전통 시장이다.",
            "1905년에 개설되었으며, 대한민국 최초의 상설 시장으로 알려져 있다.",
            "주요 판매 품목은 한복, 직물, 구제 의류, 그리고 다양한 먹거리이다.",
            "특히 빈대떡, 마약김밥, 육회 등이 유명하여 많은 관광객들이 찾는다."
        ]

    # 2. 임베딩 모델 로드
    print(f"🧠 3단계: '{EMBEDDING_MODEL}' 임베딩 모델 로드 중...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ 임베딩 모델 로드 완료.")
    except Exception as e:
        print(f"🚨 임베딩 모델 로드 실패: {e}")
        return

    # 3. 문서 임베딩
    print("⏳ 4단계: 문서 임베딩 시작 (시간이 다소 걸릴 수 있습니다)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"(사용 장치: {device})")
    doc_embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        device=device
    )
    print(f"✅ 임베딩 완료. 임베딩 벡터 형태: {doc_embeddings.shape}")

    # 4. ChromaDB에 저장
    print(f"💾 5단계: ChromaDB에 데이터 저장 시작...")
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

        # 각 문서에 대한 고유 ID 생성
        doc_ids = [str(i) for i in range(len(documents))]

        # 컬렉션에 데이터 추가 (이미 ID가 존재하면 덮어씀)
        collection.upsert(
            embeddings=doc_embeddings.tolist(),
            documents=documents,
            ids=doc_ids
        )
        print(f"✅ 저장 완료! '{COLLECTION_NAME}' 컬렉션에 총 {collection.count()}개의 문서가 저장되었습니다.")
        print(f"(DB 저장 위치: '{os.path.abspath(PERSIST_DIRECTORY)}')")

    except Exception as e:
        print(f"🚨 ChromaDB 저장 중 오류 발생: {e}")


if __name__ == '__main__':
    main()