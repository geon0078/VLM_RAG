import os
import csv
import subprocess
import re
import sys

def parse_output(output):
    """
    모델 실행 결과(stdout)에서 '[질문]' 태그를 기준으로 최종 결과 블록을 찾아
    디버깅 메시지를 제외한 최종 답변과 근거를 추출합니다.
    """
    try:
        # 최종 결과는 항상 '[질문]'으로 시작하므로, 이 부분을 기준으로 검색합니다.
        final_block_start_index = output.find('[질문]')
        
        if final_block_start_index == -1:
            return "최종 결과 블록([질문])을 찾을 수 없음", f"전체 출력에서 [질문] 태그를 찾지 못했습니다."

        # '[질문]' 태그부터 문자열 끝까지를 최종 결과 블록으로 간주합니다.
        result_block = output[final_block_start_index:]

        # 결과 블록 내에서 답변과 근거를 추출합니다.
        answer_match = re.search(r"[최종 답변]\s*(.*?)\s*[근거]", result_block, re.DOTALL)
        reasoning_match = re.search(r"[근거]\s*(.*)", result_block, re.DOTALL)

        answer = answer_match.group(1).strip() if answer_match else "답변을 찾을 수 없음"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "근거를 찾을 수 없음"
        
        # 근거 부분의 마지막 "====..." 줄이 있다면 제거합니다.
        if '===' in reasoning:
            reasoning = reasoning.split('='*20)[0].strip()

        return answer, reasoning
    except Exception as e:
        print(f"출력 파싱 중 오류 발생: {e}\n--- 원본 출력 ---\n{output}\n------------------")
        return "출력 파싱 오류", "출력 파싱 오류"

def run_model(model_version_path, image_path, question):
    """
    지정된 버전의 모델을 실행하고 출력을 반환합니다.
    """
    command = [
        sys.executable, 'main.py',
        '--image_path', image_path,
        '--question', question
    ]
    
    try:
        result = subprocess.run(
            command,
            cwd=model_version_path,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"실행 실패: {os.path.basename(model_version_path)} (이미지: {os.path.basename(image_path)})"
        print(f"  - {error_message}")
        core_error = e.stderr.strip().split('\n')[-1]
        print(f"  - 오류: {core_error}")
        return f"{error_message}\n오류: {core_error}"
    except FileNotFoundError:
        return f"실행 실패: {model_version_path} 에서 main.py를 찾을 수 없습니다."


def main():
    """
    V1과 V3 모델의 출력을 비교하여 CSV 파일로 즉시 저장하는 메인 함수
    """
    base_dir = '/home/aisw/Project/UST-ETRI-2025/VLM_RAG'
    v1_dir = os.path.join(base_dir, 'v1')
    v3_dir = os.path.join(base_dir, 'v3')
    images_dir = os.path.join(base_dir, 'images')
    questions_csv_path = os.path.join(base_dir, 'gradio', 'questions.csv')
    output_csv_path = os.path.join(base_dir, 'comparison_v1_v3.csv')

    if not os.path.exists(questions_csv_path):
        print(f"오류: 질문 CSV 파일을 찾을 수 없습니다 - {questions_csv_path}")
        return

    with open(questions_csv_path, 'r', encoding='utf-8') as f:
        questions_list = list(csv.DictReader(f))
        total_questions = len(questions_list)

    header = ['Image', 'Question', 'V1_Answer', 'V1_Reasoning_and_References', 'V3_Answer', 'V3_Reasoning_and_References']
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

    for i, row in enumerate(questions_list):
        image_name = row.get('image_name')
        question = row.get('question')
        progress = f"({i + 1}/{total_questions})"

        print(f"--- 처리 중: {image_name} {progress} ---")

        if not image_name or not question:
            print(f"  잘못된 행, 건너뜁니다.")
            continue

        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"  경고: 이미지 파일을 찾을 수 없어 건너뜁니다 - {image_path}")
            continue
        
        print(f"  [V1] 실행 중...")
        v1_output = run_model(v1_dir, image_path, question)
        v1_answer, v1_reasoning = parse_output(v1_output)
        print(f"  [V1] 완료.")

        print(f"  [V3] 실행 중...")
        v3_output = run_model(v3_dir, image_path, question)
        v3_answer, v3_reasoning = parse_output(v3_output)
        print(f"  [V3] 완료.")

        result_row = {
            'Image': image_name,
            'Question': question,
            'V1_Answer': v1_answer,
            'V1_Reasoning_and_References': v1_reasoning,
            'V3_Answer': v3_answer,
            'V3_Reasoning_and_References': v3_reasoning,
        }

        with open(output_csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(result_row)
        
        print(f"--- 저장 완료: {image_name} ---\n")

    print(f"\n비교 완료. 모든 결과가 {output_csv_path} 에 저장되었습니다.")

if __name__ == "__main__":
    main()
