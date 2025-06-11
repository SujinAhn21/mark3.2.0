# train_mark3_2_0.py
# back to basic(얼리스타핑은 해도 학습 스케줄러는 웩)  
# [12:33] custom_collate 함수 위치 조정  
# 역할: 5명의 전문가 Teacher 지식을 통합하여 Student 모델(mark3.2.0)을 학습시키는 메인 스크립트

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import argparse
import torch.nn.functional as F

# --- 기존 경로 설정 및 모듈 import (teacher_train.py와 동일) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.append(UTILS_DIR)

from vild_utils import normalize_mel_shape
from vild_config import AudioViLDConfig
from vild_model import SimpleAudioEncoder, ViLDTextHead
from vild_parser_teacher import AudioParser # Labeled/Unlabeled 모두 파싱 가능
from seed_utils import set_seed


# ==============================================================================
# 1. Ensemble Teacher 모델 클래스 정의
# (5명의 전문가 지식을 실시간으로 통합하는 핵심 로직)
# ==============================================================================
class EnsembleTeacher:
    def __init__(self, specialist_config, device):
        self.device = device
        self.specialists = {}
        self.text_embs = {}

        # 각 전문가별 설정 로드
        for class_name, paths in specialist_config.items():
            print(f"--- Loading specialist for '{class_name}' ---")
            # 각 전문가는 자신의 mark_version에 맞는 config를 가짐
            config = AudioViLDConfig(mark_version=paths['mark_version'])
            
            # Encoder와 Classifier Head 로드
            encoder = SimpleAudioEncoder(config).to(self.device)
            encoder.load_state_dict(torch.load(paths['encoder_path'], map_location=device))
            encoder.eval()

            classifier = ViLDTextHead(config).to(self.device)
            classifier.load_state_dict(torch.load(paths['classifier_path'], map_location=device))
            classifier.eval()

            self.specialists[class_name] = {'encoder': encoder, 'classifier': classifier}

            # 각 전문가가 사용하는 텍스트 프롬프트 및 임베딩 생성
            text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            classes = config.get_classes_for_text_prompts() # 예: ['thumping', 'others']
            prompt_texts = [f"A sound of {cls.replace('_', ' ')} in the room" for cls in classes]
            text_emb = torch.tensor(text_model.encode(prompt_texts), dtype=torch.float).to(device)
            self.text_embs[class_name] = text_emb
            print(f"  - Loaded. Classes: {classes}")

    @torch.no_grad()
    def __call__(self, unlabeled_audio_batch, student_class_map):
        """
        Unlabeled 오디오 배치에 대해 5명의 전문가 의견을 종합하여
        하나의 Soft Target(6차원 로짓)을 생성.
        """
        batch_size = unlabeled_audio_batch.size(0)
        # 최종 Soft Target (logits)을 저장할 텐서 (batch_size, 6)
        # 6개 클래스: thumping, water, drill, hammer, voice, others
        ensembled_logits = torch.zeros(batch_size, len(student_class_map)).to(self.device)
        
        # 각 전문가 모델의 예측을 수집
        others_logits_sum = torch.zeros(batch_size, 1).to(self.device)

        for class_name, models in self.specialists.items():
            # 전문가의 오디오 임베딩 추출
            region_emb = models['encoder'](unlabeled_audio_batch)
            # 전문가의 로짓 계산 (예: [B, 2] 크기)
            logits = models['classifier'](region_emb, self.text_embs[class_name])
            
            # 해당 클래스의 로짓을 최종 점수판에 할당
            target_idx = student_class_map[class_name]
            ensembled_logits[:, target_idx] = logits[:, 0] # 0번 인덱스가 해당 클래스 로짓이라고 가정

            # 'others' 로짓을 누적 (평균을 내기 위해)
            others_logits_sum += logits[:, 1].unsqueeze(1)

        # 'others' 점수는 5개 모델의 'others' 로짓 평균값으로 계산
        others_idx = student_class_map['others']
        ensembled_logits[:, others_idx] = (others_logits_sum / len(self.specialists)).squeeze()
        
        return ensembled_logits

# ==============================================================================
# 2. 새로운 데이터셋 클래스 (Labeled / Unlabeled 모두 처리)
# ==============================================================================
class SemiSupervisedDataset(Dataset):
    def __init__(self, file_path_list, parser, config, is_labeled=True):
        self.samples = []
        self.parser = parser
        self.config = config
        self.is_labeled = is_labeled
        
        if is_labeled:
            valid_labels = set(config.get_classes_for_text_prompts())

        for item in file_path_list:
            path = item['path']
            label = item.get('label') # Unlabeled 데이터는 label 키가 없을 수 있음
            
            if is_labeled and label not in valid_labels:
                continue

            try:
                # AudioParser는 오디오를 여러 세그먼트로 나눌 수 있음
                segments, _ = parser.parse_sample(path, label if is_labeled else "dummy_label")
                for seg in segments:
                    # normalize_mel_shape은 teacher_train.py에 없었지만, parser에 있으므로 추가
                    seg_normalized = normalize_mel_shape(seg) 
                    if seg_normalized is not None:
                        # Unlabeled 데이터의 레이블은 -1로 통일
                        final_label = label if is_labeled else -1
                        self.samples.append((seg_normalized, final_label))
            except Exception as e:
                print(f"[ERROR] Failed to parse {path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ======================================
# [수정] DataLoader를 위한 유틸리티 함수 >> 순환관계로 인한 위치조정.
# ======================================
######  custom_collate는 teacher_train.py에서 가져옴
'''
DataLoader를 위한 커스텀 collate 함수.
배치 내의 mel-spectrogram과 라벨을 분리하고 텐서로 변환하는데... 
'''

def custom_collate(batch):
    mels, labels = zip(*batch)
    mels = torch.stack(mels, dim=0)
    return mels, list(labels)


# ==============================================================================
# 3. 메인 학습 함수
# ==============================================================================
def train_mark3_2_0(seed_value=42, mark_version="mark3.2.0"):
    set_seed(seed_value)
    
    # --- 1. 설정 및 초기화 ---
    # Student 모델(mark3.2.0)을 위한 메인 설정
    config = AudioViLDConfig(mark_version=mark_version)
    device = config.device

    # 지식 증류 파라미터 (보고서 내용 반영)
    ALPHA = 0.7  # Soft Loss 가중치
    TEMPERATURE = 4.0 # Softmax 부드럽게 만들기

    # --- 2. 데이터 준비 ---
    parser = AudioParser(config, segment_mode=True)
    
    # Labeled 데이터 로드 (보고서 3의 'Labeled Set')
    labeled_csv_path = os.path.join(BASE_DIR, "dataset_labeled.csv") # 예시 파일명
    labeled_files = []
    with open(labeled_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labeled_files.append(row)
    labeled_dataset = SemiSupervisedDataset(labeled_files, parser, config, is_labeled=True)

    # Unlabeled 데이터 로드 (보고서 3의 'Unlabeled Set')
    unlabeled_csv_path = os.path.join(BASE_DIR, "dataset_unlabeled.csv") # 예시 파일명
    unlabeled_files = []
    with open(unlabeled_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            unlabeled_files.append(row)
    unlabeled_dataset = SemiSupervisedDataset(unlabeled_files, parser, config, is_labeled=False)
    
    # 두 데이터셋을 합쳐 학습 데이터로 사용
    train_dataset = ConcatDataset([labeled_dataset, unlabeled_dataset])
    
    # TODO: 실제 검증 데이터셋(Test Set)으로 val_loader 구성 필요
    # 우선 학습 데이터 일부를 임시 검증셋으로 사용
    val_size = max(1, int(0.1 * len(labeled_dataset)))
    train_size_labeled = len(labeled_dataset) - val_size
    temp_train_labeled, val_dataset = random_split(labeled_dataset, [train_size_labeled, val_size])
    
    final_train_dataset = ConcatDataset([temp_train_labeled, unlabeled_dataset])

    
    train_loader = DataLoader(final_train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)

    # --- 3. 모델 정의 ---
    # Student 모델 (6개 클래스 분류)
    student_encoder = SimpleAudioEncoder(config).to(device)
    student_classifier = ViLDTextHead(config).to(device)
    
    # Student가 사용할 텍스트 임베딩 (6개 클래스)
    text_model_student = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    student_classes = config.get_classes_for_text_prompts()
    student_prompt_texts = [f"A sound of {cls.replace('_', ' ')} in the room" for cls in student_classes]
    student_text_emb = torch.tensor(text_model_student.encode(student_prompt_texts), dtype=torch.float).to(device)
    student_label_map = config.get_target_label_map()
    
    # Ensemble Teacher 모델
    # 수정된 코드 (train_mark3_2_0.py 내부)
    specialist_config = {
        'thumping': {
            'mark_version': 'mark2.5.0',
            'encoder_path': 'best_teacher_encoder_mark2.5.0.pth',
            'classifier_path': 'best_teacher_classifier_mark2.5.0.pth'
        },
        'water_sound': {
            'mark_version': 'mark2.6.0',
            'encoder_path': 'best_teacher_encoder_mark2.6.0.pth',
            'classifier_path': 'best_teacher_classifier_mark2.6.0.pth'
        },
        'drill_sound': {
            'mark_version': 'mark2.7.0',
            'encoder_path': 'best_teacher_encoder_mark2.7.0.pth',
            'classifier_path': 'best_teacher_classifier_mark2.7.0.pth'
        },
        'hammer_sound': {
            'mark_version': 'mark2.8.0',
            'encoder_path': 'best_teacher_encoder_mark2.8.0.pth',
            'classifier_path': 'best_teacher_classifier_mark2.8.0.pth'
        },
        'human_voice': {
            'mark_version': 'mark2.9.0',
            'encoder_path': 'best_teacher_encoder_mark2.9.0.pth',
            'classifier_path': 'best_teacher_classifier_mark2.9.0.pth'
        },
    }
    ensemble_teacher = EnsembleTeacher(specialist_config, device)

    # --- 4. 손실 함수 및 옵티마이저 ---
    hard_loss_fn = nn.CrossEntropyLoss()
    soft_loss_fn = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(list(student_encoder.parameters()) + list(student_classifier.parameters()), lr=config.learning_rate)

    # --- 5. 학습 루프 ---
    print(f"[INFO] Student training (mark3.2.0) started on {device}")
    for epoch in range(config.num_epochs):
        student_encoder.train()
        student_classifier.train()
        total_loss, total_hard_loss, total_soft_loss = 0, 0, 0
        
        for mel_batch, label_batch in train_loader:
            mel = mel_batch.to(device)
            # label_batch는 숫자(-1 포함)와 문자열이 섞인 리스트
            
            optimizer.zero_grad()
            
            # Labeled / Unlabeled 데이터 분리
            labeled_indices = [i for i, lbl in enumerate(label_batch) if isinstance(lbl, str)]
            unlabeled_indices = [i for i, lbl in enumerate(label_batch) if lbl == -1]

            loss = 0
            
            # (A) Hard Loss 계산 (Labeled 데이터에 대해)
            if labeled_indices:
                labeled_mel = mel[labeled_indices]
                labeled_labels_str = [label_batch[i] for i in labeled_indices]
                labeled_targets = torch.tensor([student_label_map[lbl] for lbl in labeled_labels_str], dtype=torch.long).to(device)

                student_region_emb = student_encoder(labeled_mel)
                student_logits = student_classifier(student_region_emb, student_text_emb)
                
                hard_loss = hard_loss_fn(student_logits, labeled_targets)
                loss += (1 - ALPHA) * hard_loss
                total_hard_loss += hard_loss.item()
            
            # (B) Soft Loss 계산 (Unlabeled 데이터에 대해)
            if unlabeled_indices:
                unlabeled_mel = mel[unlabeled_indices]
                
                # Teacher의 Soft Target 생성 (Logits 형태)
                teacher_logits = ensemble_teacher(unlabeled_mel, student_label_map)
                
                # Student의 예측
                student_region_emb_unlabeled = student_encoder(unlabeled_mel)
                student_logits_unlabeled = student_classifier(student_region_emb_unlabeled, student_text_emb)
                
                # KLDivLoss 계산
                soft_loss = soft_loss_fn(
                    F.log_softmax(student_logits_unlabeled / TEMPERATURE, dim=1),
                    F.softmax(teacher_logits / TEMPERATURE, dim=1)
                ) * (TEMPERATURE ** 2) # 손실 스케일 보정
                
                loss += ALPHA * soft_loss
                total_soft_loss += soft_loss.item()

            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_hard = total_hard_loss / len(train_loader)
        avg_soft = total_soft_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Total Loss: {avg_loss:.4f} (Hard: {avg_hard:.4f}, Soft: {avg_soft:.4f})")
        
        # (이하 검증 및 모델 저장 로직은 teacher_train.py와 유사하게 구현)
        # ...

    print("Training finished.")
    # 최종 모델 저장
    torch.save({
        "encoder_state_dict": student_encoder.state_dict(),
        "classifier_state_dict": student_classifier.state_dict()
    }, f"student_model_{mark_version}.pth")


if __name__ == "__main__":
    # parser 설정은 필요에 따라 추가
    train_mark3_2_0(seed_value=42, mark_version="mark3.2.0")