import os
import csv
import subprocess
import re
import sys

def parse_output(output):
    """
    모델 실행 결과(stdout)에서 최종 답변과 근거를 추출합니다.
    """
    try:
        # DOTALL 플래그를 사용하여 여러 줄에 걸친 내용도 매칭
        answer_match = re.search(r"[최종 답변]\s*(.*?)\s*[근거]", output, re.DOTALL)
        reasoning_match = re.search(r"[근거]\s*(.*)", output, re.DOTALL)

        answer = answer_match.group(1).strip() if answer_match else "Not found"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Not found"
        
        # 마지막 "====..." 줄 제거
        reasoning = reasoning.split('='*50)[0].strip()

        return answer, reasoning
    except Exception as e:
        print(f"Error parsing output: {e}\nOutput was:\n{output}")
        return "Error parsing", "Error parsing"

def run_model(model_version_path, image_path, question):
    """
    지정된 버전의 모델을 실행하고 출력을 반환합니다.
    """
    command = [
        sys.executable,
        'main.py',
        '--image_path',
        image_path,
        '--question',
        question
    ]
    
    print(f"Running command: {' '.join(command)} in {model_version_path}")

    try:
        result = subprocess.run(
            command,
            cwd=model_version_path,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        print(f"Successfully ran for {os.path.basename(image_path)} on {os.path.basename(model_version_path)}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running model in {model_version_path} for image {image_path}.")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return f"Execution failed: {e.stderr}"
    except FileNotFoundError:
        print(f"Error: main.py not found in {model_version_path}")
        return "Execution failed: main.py not found"


def main():
    """
    V1과 V3 모델의 출력을 비교하여 CSV 파일로 저장하는 메인 함수
    """
    base_dir = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG'
    v1_dir = os.path.join(base_dir, 'v1')
    v3_dir = os.path.join(base_dir, 'v3')
    images_dir = os.path.join(base_dir, 'images')
    questions_csv_path = os.path.join(base_dir, 'gradio', 'questions.csv')
    output_csv_path = os.path.join(base_dir, 'comparison_v1_v3.csv')

    results = []

    if not os.path.exists(questions_csv_path):
        print(f"Error: Questions CSV file not found at {questions_csv_path}")
        return

    with open(questions_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # BOM(Byte Order Mark)이 있는 경우 필드 이름에서 제거
        reader.fieldnames = [field.lstrip('\ufeff') for field in reader.fieldnames]
        
        for row in reader:
            image_name = row.get('image_name') # 'image' -> 'image_name'으로 수정
            question = row.get('question')

            if not image_name or not question:
                print(f"Skipping invalid row: {row}")
                continue

            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found, skipping: {image_path}")
                continue

            print(f"--- Processing: {image_name} ---")
            
            # Run V1
            print("\n[Running V1]")
            v1_output = run_model(v1_dir, image_path, question)
            v1_answer, v1_reasoning = parse_output(v1_output)

            # Run V3
            print("\n[Running V3]")
            v3_output = run_model(v3_dir, image_path, question)
            v3_answer, v3_reasoning = parse_output(v3_output)

            results.append({
                'Image': image_name,
                'Question': question,
                'V1_Answer': v1_answer,
                'V1_Reasoning_and_References': v1_reasoning,
                'V3_Answer': v3_answer,
                'V3_Reasoning_and_References': v3_reasoning,
            })
            print(f"--- Finished: {image_name} ---\n")


    # 결과를 CSV 파일에 저장
    header = ['Image', 'Question', 'V1_Answer', 'V1_Reasoning_and_References', 'V3_Answer', 'V3_Reasoning_and_References']
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nComparison finished. Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()
