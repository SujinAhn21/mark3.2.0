# run_pipeline_mark3.py (mark3.2.0 전용 파이프라인)

import os
import subprocess
import logging
from datetime import datetime
import argparse

# ===== 파라미터 설정 =====
parser = argparse.ArgumentParser(description="mark3.2.0 전체 학습 파이프라인 (앙상블 지식 증류)")
parser.add_argument("--mark_version", type=str, default="mark3.2.0", help="실행할 모델 버전")
args = parser.parse_args()
mark_version = args.mark_version

# ===== 경로 및 로그 설정 (기존과 동일) =====
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "logFiles")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"run_pipeline_{mark_version}.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ===== 데코레이터 및 서브프로세스 실행 함수 (기존과 동일) =====
def timed_step(func):
    def wrapper(*args, **kwargs):
        step_name = func.__name__.replace("run_", "").replace("_", " ").title()
        logging.info(f"\n[실행 시작] --> {step_name}")
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        duration = (end - start).total_seconds()
        logging.info(f"[완료] --> {step_name} (소요시간: {duration:.2f}초)")
        return result
    return wrapper

def run_subprocess(command_list):
    try:
        logging.info(f"[CMD] {' '.join(command_list)}")
        result = subprocess.run(
            command_list, capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.stdout: logging.info("[STDOUT]\n" + result.stdout)
        if result.stderr: logging.info("[STDERR]\n" + result.stderr)
        return result.returncode
    except Exception as e:
        logging.error(f"[ERROR] Subprocess 실행 중 예외 발생: {e}")
        return 1

# ===== mark3.2.0 단계별 실행 함수 정의 =====
@timed_step
def run_step0_preprocess_labeled_data():
    """Labeled 원본 오디오를 고정된 길이로 전처리합니다."""
    return run_subprocess([
        "python", "fix_audio_length_to_240000.py",
        "--input_dir", "data_source_labeled",
        "--output_dir", "data_labeled"
    ])

@timed_step
def run_step1_preprocess_unlabeled_data():
    """Unlabeled 원본 오디오를 고정된 길이로 전처리합니다."""
    return run_subprocess([
        "python", "fix_audio_length_to_240000.py",
        "--input_dir", "data_source_unlabeled",
        "--output_dir", "data_unlabeled"
    ])

@timed_step
def run_step2_generate_dataset_index():
    """학습에 사용할 Labeled/Unlabeled 데이터셋 인덱스 CSV 파일을 생성합니다."""
    # generate_dataset_index_mark3.py는 mark_version 인자만 받으면 됨
    return run_subprocess(["python", "generate_dataset_index_mark3.py", 
                           "--mode", "train", "--mark_version", mark_version])

@timed_step
def run_step3_student_model_train():
    """5명의 전문가 Teacher 지식을 활용하여 Student 모델(mark3.2.0)을 학습시킵니다."""
    # train_mark3_2_0.py는 mark_version 인자만 받으면 됨
    return run_subprocess(["python", "train_mark3_2_0.py", "--mark_version", mark_version])

# 테스트 단계로 넘어옴  
@timed_step
def run_step_4_preprocess_test_data():
    """Test 원본 오디오를 고정된 길이로 전처리합니다."""
    return run_subprocess([
        "python", "fix_audio_length_to_240000.py",
        "--input_dir", "data_source_test",
        "--output_dir", "data_test"
    ])

@timed_step
def run_step_5_generate_test_index():
    """평가에 사용할 Test 데이터셋 인덱스 CSV 파일을 생성합니다."""
    return run_subprocess(["python", "generate_test_index_mark3.py", 
                           "--mode", "test", "--mark_version", mark_version])

@timed_step
def run_step6_evaluate_model():
    """학습된 Student 모델(C) 및 모든 비교 모델델의 성능을 평가합니다. (test_set 사용)"""
    """비교 모델: A, B, C """
    # 모델 C (mark3.2.0) 평가
    return run_subprocess(["python", "eval.py", "--mark_version", mark_version])

    
    


# ===== 메인 실행 =====
if __name__ == "__main__":
    logging.info("="*50)
    logging.info(f"  mark3.2.0 전체 학습 파이프라인 (앙상블 지식 증류) 시작  ")
    logging.info("="*50)
    logging.info(f"모델 버전: {mark_version}")
    logging.info("기존 전문가 모델들의 지식을 통합하여 새로운 Student 모델을 학습합니다.")

    # mark3.2.0 파이프라인 단계
    steps = [
        run_step0_preprocess_labeled_data,
        run_step1_preprocess_unlabeled_data,
        run_step2_generate_dataset_index,
        run_step3_student_model_train,
        run_step_4_preprocess_test_data,
        run_step_5_generate_test_index,
        run_step6_evaluate_model
    ]

    for step in steps:
        return_code = step()
        if return_code != 0:
            logging.error(f"\n[CRITICAL ERROR] 파이프라인 실패: '{step.__name__}' 단계에서 오류 발생 (종료 코드: {return_code}).")
            logging.error("이후 단계를 생략하고 파이프라인을 중단합니다.")
            break
    
    logging.info("="*50)
    logging.info(f"[종료] 전체 파이프라인 완료. 로그 파일: {log_file_path}")
    logging.info("="*50)