
# /home/aisw/Project/UST-ETRI-2025/VLM_RAG/make_vector_db_/make_vector_db_v2.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import chromadb
import os
from tqdm import tqdm

# --- 설정부 (Configuration) ---
EXCEL_FILE_PATH = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG/data/korean_train_20250101.xlsx'
EMBEDDING_MODEL = 'BAAI/bge-m3'
PERSIST_DIRECTORY = '/home/aisw/Project/UST-ETRI-2025/chroma_db'
# 새로운 컬렉션 이름
NEW_COLLECTION_NAME = "korean_knowledge_base_v3"

def main():
    """
    Excel 파일로부터 데이터를 읽어, 각 속성을 개별 문서로 분해하고
    메타데이터를 부여하여 ChromaDB 벡터 데이터베이스를 생성합니다.
    """
    # 1. 데이터 로드
    print(f"📄 1단계: '{EXCEL_FILE_PATH}' 엑셀 파일 읽기 시작...")
    try:
        df = pd.read_excel(EXCEL_FILE_PATH)
        # 불필요한 컬럼 및 결측치가 많은 컬럼 제외
        df = df.drop(columns=['역명(영어)', '역명(로마자)', '역명(일본어)', '역명(중국어간체)', '역명(중국어번체)', '역명(부역명)', '폐지일자', '참고사항', 'Unnamed: 29'], errors='ignore')
        # 역명(한글)이 없는 행은 의미가 없으므로 제거
        df.dropna(subset=['역명(한글)'], inplace=True)
        print("✅ 엑셀 파일 읽기 및 기본 전처리 완료.")
    except FileNotFoundError:
        print(f"🚨 오류: '{EXCEL_FILE_PATH}' 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"🚨 엑셀 파일 처리 중 오류 발생: {e}")
        return

    # 2. 문서 및 메타데이터 생성
    print("🔄 2단계: 원자적 문서 및 메타데이터 생성 중...")
    documents = []
    metadatas = []
    ids = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="[Processing Rows]"):
        station_name = row['역명(한글)']
        
        for col_name, value in row.items():
            # 결측치(NaN)이거나 내용이 없는 값은 건너뛰기
            if pd.isna(value) or str(value).strip() == "":
                continue
            
            # 각 속성을 하나의 완전한 문장으로 구성
            # 예: "검암역의 운영노선은 공항철도선입니다."
            document_text = f"{station_name}의 {col_name}은(는) {value}입니다."
            
            documents.append(document_text)
            metadatas.append({'역명': station_name, '속성': col_name})
            ids.append(f"{station_name}_{col_name}_{index}")

    print(f"✅ 총 {len(documents)}개의 원자적 문서를 생성했습니다.")

    # 3. 임베딩 모델 로드
    print(f"🧠 3단계: '{EMBEDDING_MODEL}' 임베딩 모델 로드 중...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ 임베딩 모델 로드 완료.")
    except Exception as e:
        print(f"🚨 임베딩 모델 로드 실패: {e}")
        return

    # 4. 문서 임베딩
    print("⏳ 4단계: 문서 임베딩 시작 (시간이 다소 걸릴 수 있습니다)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"(사용 장치: {device})")
    doc_embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True,
        device=device
    )
    print(f"✅ 임베딩 완료. 임베딩 벡터 형태: {doc_embeddings.shape}")

    # 5. ChromaDB에 저장
    print(f"💾 5단계: ChromaDB에 데이터 저장 시작...")
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        # 기존 컬렉션이 있다면 삭제하고 새로 생성
        if NEW_COLLECTION_NAME in [c.name for c in client.list_collections()]:
            client.delete_collection(name=NEW_COLLECTION_NAME)
            print(f"💥 기존 '{NEW_COLLECTION_NAME}' 컬렉션을 삭제했습니다.")
            
        collection = client.create_collection(
            name=NEW_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용
        )

        # 데이터를 5000개씩 나누어 저장 (메모리 관리)
        batch_size = 5000
        for i in tqdm(range(0, len(documents), batch_size), desc="[Saving to DB]"):
            collection.add(
                embeddings=doc_embeddings[i:i+batch_size].tolist(),
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
            
        print(f"✅ 저장 완료! '{NEW_COLLECTION_NAME}' 컬렉션에 총 {collection.count()}개의 문서가 저장되었습니다.")
        print(f"(DB 저장 위치: '{os.path.abspath(PERSIST_DIRECTORY)}')")

    except Exception as e:
        print(f"🚨 ChromaDB 저장 중 오류 발생: {e}")

if __name__ == '__main__':
    main()
