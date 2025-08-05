
# /home/aisw/Project/UST-ETRI-2025/VLM_RAG/make_vector_db_/line_data_v2.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import chromadb
from tqdm import tqdm

# --- 설정부 ---
DATA_FOLDER = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG/data/csv'
EMBEDDING_MODEL = 'BAAI/bge-m3'
PERSIST_DIRECTORY = '/home/aisw/Project/UST-ETRI-2025/chroma_db'
# 새로운 컬렉션 이름
NEW_COLLECTION_NAME = 'subway_line_info_v1'

def load_csv_files(folder_path):
    """지정된 폴더에서 .csv 파일 목록을 불러옵니다."""
    csv_files = []
    for filename in os.listdir(folder_path):
        # 'convert'가 포함된 파일은 변환용 스크립트이므로 제외
        if filename.lower().endswith('.csv') and 'convert' not in filename:
            csv_files.append(os.path.join(folder_path, filename))
    return csv_files

def build_atomic_documents_from_csv(file_path):
    """
    CSV 파일에서 역 순서 정보를 읽어,
    역과 역 사이의 관계를 나타내는 원자적 문서와 메타데이터를 생성합니다.
    """
    line_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)
    # '역명(한글)' 컬럼에 결측치가 있을 경우 제거하고 리스트로 변환
    station_names = df['역명(한글)'].dropna().tolist()
    
    documents = []
    metadatas = []
    ids = []

    # 역 리스트를 순회하며 "이전-다음" 관계의 문서를 생성
    for i in range(len(station_names) - 1):
        prev_station = station_names[i]
        next_station = station_names[i+1]

        # 다음 역 정보 문서
        doc_next = f"{line_name}에서 {prev_station}의 다음 역은 {next_station}입니다."
        meta_next = {'노선': line_name, '역': prev_station, '방향': '다음'}
        id_next = f"{line_name}_{prev_station}_next"
        documents.append(doc_next)
        metadatas.append(meta_next)
        ids.append(id_next)

        # 이전 역 정보 문서
        doc_prev = f"{line_name}에서 {next_station}의 이전 역은 {prev_station}입니다."
        meta_prev = {'노선': line_name, '역': next_station, '방향': '이전'}
        id_prev = f"{line_name}_{next_station}_prev"
        documents.append(doc_prev)
        metadatas.append(meta_prev)
        ids.append(id_prev)
        
    return documents, metadatas, ids

def main():
    print("📂 CSV 파일 탐색 중...")
    csv_files = load_csv_files(DATA_FOLDER)
    if not csv_files:
        print("⚠️ CSV 파일이 없습니다.")
        return
    print(f"✅ 총 {len(csv_files)}개 노선 CSV 파일 발견.")
    
    all_documents = []
    all_metadatas = []
    all_ids = []

    for csv_file in tqdm(csv_files, desc="[Processing Lines]"):
        try:
            docs, metas, ids = build_atomic_documents_from_csv(csv_file)
            all_documents.extend(docs)
            all_metadatas.extend(metas)
            all_ids.extend(ids)
        except Exception as e:
            line_name = os.path.splitext(os.path.basename(csv_file))[0]
            print(f"🚨 '{line_name}' 처리 중 오류: {e}")

    print(f"✅ 총 {len(all_documents)}개의 역 관계 문서를 생성했습니다.")

    print(f"🧠 임베딩 모델 로딩: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"⏳ 임베딩 중... (디바이스: {device})")
    embeddings = model.encode(all_documents, show_progress_bar=True, device=device)

    print("💾 ChromaDB 저장 시작...")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # 기존 컬렉션이 있다면 삭제하고 새로 생성
    if NEW_COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=NEW_COLLECTION_NAME)
        print(f"💥 기존 '{NEW_COLLECTION_NAME}' 컬렉션을 삭제했습니다.")
        
    collection = client.create_collection(
        name=NEW_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 데이터를 배치로 나누어 저장
    batch_size = 5000
    for i in tqdm(range(0, len(all_documents), batch_size), desc="[Saving to DB]"):
        collection.add(
            embeddings=embeddings[i:i+batch_size].tolist(),
            documents=all_documents[i:i+batch_size],
            metadatas=all_metadatas[i:i+batch_size],
            ids=all_ids[i:i+batch_size]
        )

    print(f"✅ 저장 완료: {collection.count()}건")
    print(f"(DB 위치: {os.path.abspath(PERSIST_DIRECTORY)})")

if __name__ == '__main__':
    main()
