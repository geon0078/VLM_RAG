import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import chromadb

# --- 설정부 ---
DATA_FOLDER = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG/data/csv'
EMBEDDING_MODEL = 'BAAI/bge-m3'
# ChromaDB를 저장할 디렉터리 경로
PERSIST_DIRECTORY = '/home/aisw/Project/UST-ETRI-2025/chroma_db'
# 저장할 컬렉션 이름 (main.py와 동일해야 함)
COLLECTION_NAME = 'subway_multilang_all_lines'

def load_csv_files(folder_path):
    csv_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            csv_files.append(os.path.join(folder_path, filename))
    return csv_files

def build_documents_from_csv(file_path, file_key):
    df = pd.read_csv(file_path)
    documents = []
    ids = []
    # 역명(한글) 컬럼만 추출하여 순서대로 리스트로 저장
    station_names = df['역명(한글)'].tolist()
    # 한 노선의 모든 역 순서를 하나의 문서로 저장
    text = f"지하철 노선: {file_key} / 역 순서: {' -> '.join(station_names)}"
    documents.append(text)
    ids.append(f"{file_key}_all")
    return documents, ids

def main():
    print("📂 CSV 파일 탐색 중...")
    csv_files = load_csv_files(DATA_FOLDER)
    if not csv_files:
        print("⚠️ CSV 파일이 없습니다.")
        return

    print(f"✅ 총 {len(csv_files)}개 CSV 파일 발견.")
    
    all_documents = []
    all_ids = []

    for csv_file in csv_files:
        file_key = os.path.splitext(os.path.basename(csv_file))[0]
        try:
            docs, ids = build_documents_from_csv(csv_file, file_key)
            all_documents.extend(docs)
            all_ids.extend(ids)
            print(f"📄 '{file_key}' 처리 완료. {len(docs)}개 문장 생성.")
        except Exception as e:
            print(f"🚨 '{file_key}' 처리 중 오류: {e}")

    print(f"🧠 임베딩 모델 로딩: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"⏳ 임베딩 중... 총 문서 수: {len(all_documents)} (디바이스: {device})")
    embeddings = model.encode(all_documents, show_progress_bar=True, device=device)

    print("💾 ChromaDB 저장 시작...")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    collection.upsert(ids=all_ids, documents=all_documents, embeddings=embeddings.tolist())

    print(f"✅ 저장 완료: {collection.count()}건")
    print(f"(DB 위치: {os.path.abspath(PERSIST_DIRECTORY)})")

if __name__ == '__main__':
    main()
