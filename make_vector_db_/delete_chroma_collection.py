import chromadb
import os

# 설정: 삭제할 컬렉션 이름과 DB 경로를 config와 동일하게 맞추세요
PERSIST_DIRECTORY = '../chroma_db'
COLLECTION_NAME = 'subway_multilang_all_lines'

if __name__ == '__main__':
    print(f"ChromaDB 경로: {os.path.abspath(PERSIST_DIRECTORY)}")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"✅ 컬렉션 '{COLLECTION_NAME}' 삭제 완료.")
    except Exception as e:
        print(f"🚨 컬렉션 삭제 실패: {e}")
