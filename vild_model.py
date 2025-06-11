# vild_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAudioEncoder(nn.Module):
    """
    Mel-Spectrogram 입력을 고정된 차원의 임베딩으로 변환하는 CNN 기반 오디오 인코더

    특징:
    - Dropout 추가로 과적합 방지
    - AdaptiveAvgPool + LayerNorm 구조로 소형 모델에서도 안정성 확보
    - 최소 입력 크기 추론을 위한 get_min_input_shape() 제공
    """

    def __init__(self, config):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
            nn.Linear(64, config.embedding_dim)
        )

        self.model = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.head
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, 1, 64, 101] 형태의 mel spectrogram

        Returns:
            Tensor: [B, embedding_dim] 형태의 오디오 임베딩 벡터
        """
        return self.model(x)

    @staticmethod
    def get_min_input_shape(config=None):
        """
        모델 구조를 기반으로 한 최소 입력 크기 반환

        Returns:
            Tuple[int, int, int, int]: (B, C, H, W)
        """
        h = getattr(config, 'n_mels', 64) if config else 64
        w = getattr(config, 'min_time_frames', 101) if config else 101
        return (1, 1, h, w)


class ViLDTextHead(nn.Module):
    """
    region embedding과 텍스트 임베딩 간 cosine similarity 로짓을 계산하는 헤드 모듈

    특징:
    - Temperature scaling 적용
    - background class는 사용하지 않음 (ex. binary 분류나 multi-class 분류 시 직접 사용)
    - CrossEntropyLoss 또는 soft label 학습 시 softmax 적용은 외부에서 처리
    """

    def __init__(self, config):
        super().__init__()
        self.temperature = getattr(config, 'logit_temperature', 0.07)

    def forward(self, region_embeddings, class_text_embeddings):
        """
        Args:
            region_embeddings (Tensor): [B, D], student or teacher region embeddings
            class_text_embeddings (Tensor): [C, D], 사전 학습된 텍스트 임베딩

        Returns:
            Tensor: [B, C] 형태의 로짓 벡터 (softmax 적용 전)
        """
        region_norm = F.normalize(region_embeddings, dim=1)
        text_norm = F.normalize(class_text_embeddings, dim=1)

        logits = torch.matmul(region_norm, text_norm.T)  # [B, C]
        logits = logits / self.temperature
        return logits
    