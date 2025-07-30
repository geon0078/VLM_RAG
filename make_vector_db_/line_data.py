import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import chromadb

# --- 설정부 ---
DATA_FOLDER = '../data/csv'
EMBEDDING_MODEL = 'BAAI/bge-m3'
# ChromaDB를 저장할 디렉터리 경로
PERSIST_DIRECTORY = '../chroma_db'
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
    for idx, row in df.iterrows():
        text = (
            f"지하철 노선: {file_key} / 역명: {row['역명(한글)']} / "
            f"영문: {row['역명(영문)']} / 로마자: {row['역명(로마자)']} / "
            f"일본어: {row['역명(일본어)']} / 중국어(간체): {row['역명(중국어 간체)']} / "
            f"중국어(번체): {row['역명(중국어 번체)']}"
        )
        documents.append(text)
        ids.append(f"{file_key}_{idx}")
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
