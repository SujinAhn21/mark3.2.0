# eval.py (mark3.2.0)
# vild_config.py 수정내용 반영... 
# [12:43] 

import os
import sys
import csv
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 경로 설정 및 모듈 import ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.append(UTILS_DIR)

# 필요한 모듈 import
from train_mark3_2_0 import SemiSupervisedDataset, custom_collate
from vild_model import SimpleAudioEncoder, ViLDTextHead
from vild_config import AudioViLDConfig # 수정된 config 사용
from vild_parser_teacher import AudioParser
# SentenceTransformer는 config 내부에서 호출되므로 여기서 직접 사용할 필요 없음

def evaluate(mark_version: str):
    print("="*60)
    print(f"모델 성능 평가 시작: {mark_version}")
    print("="*60)

    # --- 1. 설정 및 모델, 데이터 로드 ---
    config = AudioViLDConfig(mark_version=mark_version)
    device = config.device
    
    # 평가용 데이터셋 로드 (dataset_test.csv 사용 가정)
    test_csv_path = os.path.join(BASE_DIR, "dataset_test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"[ERROR] 테스트 인덱스 파일을 찾을 수 없습니다: {test_csv_path}")

    test_files = []
    with open(test_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_files.append(row)

    parser = AudioParser(config, segment_mode=True)
    # is_labeled=True로 설정하여 레이블이 있는 테스트셋을 로드
    test_dataset = SemiSupervisedDataset(test_files, parser, config, is_labeled=True)  
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate)

    # 학습된 Student 모델 로드
    model_path = f"student_model_{mark_version}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] 학습된 모델 파일을 찾을 수 없습니다: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    student_encoder = SimpleAudioEncoder(config).to(device)
    student_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    student_encoder.eval()

    student_classifier = ViLDTextHead(config).to(device)
    student_classifier.load_state_dict(checkpoint['classifier_state_dict'])
    student_classifier.eval()
    
    # [수정] 평가에 사용될 클래스 관련 정보를 새로운 메서드를 통해 일관되게 가져옵니다.
    # 1. 평가용 클래스 이름 (6개)
    class_names = config.get_classes_for_evaluation()
    # 2. 평가용 텍스트 임베딩 (6개 클래스에 대한 임베딩)
    student_text_emb = config.get_class_text_embeddings(for_evaluation=True)
    # 3. 평가용 라벨 맵 (6개 클래스에 대한 맵)
    student_label_map = config.get_target_label_map()


    # --- 2. 추론 및 결과 수집 ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mel_batch, label_batch_str in test_loader:
            mel = mel_batch.to(device)
            # 예측
            region_emb = student_encoder(mel)
            logits = student_classifier(region_emb, student_text_emb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            # label_batch_str의 각 레이블을 student_label_map을 통해 정수 인덱스로 변환
            all_labels.extend([student_label_map[lbl] for lbl in label_batch_str])

    # --- 3. 평가 지표 계산 및 출력 ---
    print("\n" + "-"*20 + " 종합 성능 평가 " + "-"*20)

    # [수정] classification_report에 전달되는 target_names는 위에서 정의한 class_names를 사용합니다.
    # 이 class_names는 이제 6개의 항목을 가지므로 오류가 발생하지 않습니다.
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("\n[분류 리포트 (Classification Report)]")
    print(report)

    # (2) Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n[전체 정확도 (Overall Accuracy)]")
    print(f"{accuracy:.4f} ({accuracy*100:.2f}%)")

    # (3) Confusion Matrix
    print("\n[혼동 행렬 (Confusion Matrix)]")
    # [수정] 혼동 행렬의 인덱스, 컬럼명도 class_names를 사용합니다.
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    # --- 4. 혼동 행렬 시각화 및 저장 ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 12})
    plt.title(f'Confusion Matrix - {mark_version}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f"confusion_matrix_{mark_version}.png")
    plt.savefig(save_path)
    print(f"\n[시각화] 혼동 행렬 이미지가 저장되었습니다: {save_path}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="학습된 모델의 성능을 평가합니다.")
    parser.add_argument('--mark_version', type=str, required=True, help="평가할 모델의 버전 (예: mark3.2.0)")
    args = parser.parse_args()
    
    evaluate(mark_version=args.mark_version)  