import os
import pandas as pd


# 변환할 .xls 파일들이 있는 폴더 경로 (지정하지 않으면 현재 작업 디렉토리)
FOLDER_PATH = os.getcwd()

def convert_xls_to_csv(folder_path):
    trash_dir = os.path.join(folder_path, 'trash')
    os.makedirs(trash_dir, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.xls'):
            xls_path = os.path.join(folder_path, filename)
            csv_filename = os.path.splitext(filename)[0] + '.csv'
            csv_path = os.path.join(folder_path, csv_filename)

            try:
                # xlrd 엔진을 이용하여 .xls 파일 읽기
                df = pd.read_excel(xls_path, engine='xlrd', header=None)
                df.to_csv(csv_path, index=False, header=False)
                print(f"✅ 변환 완료: {filename} → {csv_filename}")
                # 변환 후 trash 폴더로 이동
                os.rename(xls_path, os.path.join(trash_dir, filename))
                print(f"🗑️ {filename} → trash 폴더로 이동 완료")
            except Exception as e:
                print(f"⚠️ 변환 실패: {filename} → {e}")

if __name__ == '__main__':
    convert_xls_to_csv(FOLDER_PATH)