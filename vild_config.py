# vild_config.py
# 뭐가 많아질수록 힘들구나.  
# [12:41]

import torch
from sentence_transformers import SentenceTransformer
import os

class AudioViLDConfig:
    def __init__(self, mark_version="mark3.2.0"):
        self.mark_version = mark_version

        # ==============================================================================
        # 1. 클래스 설정 (Mark Version 별)
        # ==============================================================================
        if self.mark_version == "mark2.5.0":
            self.classes = ["thumping", "others"]
        elif self.mark_version == "mark2.6.0":
            # 이전 로그에서 'water_sound' 대신 'water'를 사용하는 것을 확인. 일관성을 위해 수정.
            self.classes = ["water_sound", "others"] 
        elif self.mark_version == "mark2.7.0":
            # 이전 로그에서 'drill_sound' 대신 'drill'을 사용하는 것을 확인. 일관성을 위해 수정.
            self.classes = ["drill_sound", "others"] 
        elif self.mark_version == "mark2.8.0":
             # 이전 로그에서 'hammer_sound' 대신 'hammer'을 사용하는 것을 확인. 일관성을 위해 수정.
            self.classes = ["hammer_sound", "others"]
        elif self.mark_version == "mark2.9.0":
            # 이전 로그에서 'human_voice' 대신 'voice'를 사용하는 것을 확인. 일관성을 위해 수정.
            self.classes = ["human_voice", "others"]
        elif self.mark_version == "mark3.2.0":
            self.classes = [
                "thumping", "water_sound", "drill_sound",
                "hammer_sound", "human_voice", "others", 
                "dummy_label" # vild_parser_teacher가 unlabeled 데이터를 처리하기 위한 임시 레이블
            ]
        else:
            raise ValueError(
                f"[Error] Unknown or unsupported mark_version: '{self.mark_version}'.\n"
                f"지원되는 값: ['mark2.5.0', 'mark2.6.0', 'mark2.7.0', 'mark2.8.0', 'mark2.9.0', 'mark3.2.0']"
            )

        # === 기존 속성 유지 ===
        # labeled_classes는 이제 파서가 사용할 모든 클래스를 의미하게 됨
        self.labeled_classes = self.classes
        self.unlabeled_class_identifier = "unlabeled"
        self.num_distinct_labeled_classes = len(self.labeled_classes)


        # ==============================================================================
        # 2. 오디오 및 모델 공통 파라미터
        # ==============================================================================
        # === 오디오 파라미터 ===
        self.sample_rate = 16000
        self.segment_duration = 1.0
        self.segment_samples = int(self.sample_rate * self.segment_duration)
        self.fft_size = 1024
        self.hop_length = 160
        self.n_mels = 64

        # === Segment 단위 처리 ===
        self.segment_length = 101   # Mel spectrogram time frame 수
        self.segment_hop = 50       # Segment 간 stride
        self.max_segments = 5       # Teacher가 사용할 최대 segment 수

        # === 모델 파라미터 ===
        self.embedding_dim = 384
        self.use_background_embedding = True

        # === 학습 파라미터 ===
        self.batch_size = 16
        self.num_epochs = 80
        self.learning_rate = 1e-4
        self.text_loss_weight = 1.0
        self.image_loss_weight = 1.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # === 데이터 경로 ===
        self.audio_dir = os.path.join("data_wav")

        # === 내부 캐시 ===
        self._text_emb = None
        self._eval_text_emb = None # 평가용 텍스트 임베딩 캐시 추가


    # ==============================================================================
    # 3. 클래스 및 텍스트 관련 메서드
    # ==============================================================================
    def get_class_index(self, class_name: str) -> int:
        """주어진 클래스 이름의 인덱스를 반환합니다. Unlabeled의 경우 -1을 반환합니다."""
        if class_name in self.labeled_classes:
            return self.labeled_classes.index(class_name)
        elif class_name == self.unlabeled_class_identifier:
            return -1
        else:
            raise ValueError(
                f"[Config Error] '{class_name}'는 mark_version '{self.mark_version}'에 등록되지 않은 클래스입니다.\n"
                f"=> 현재 사용 가능한 클래스: {self.labeled_classes}"
            )

    def get_classes_for_parser(self) -> list:
        """
        데이터 파싱 시 유효한 모든 레이블 목록을 반환합니다. (dummy_label 포함)
        학습 데이터셋 구성 시 사용됩니다.
        """
        return self.labeled_classes

    def get_classes_for_text_prompts(self) -> list:
        """
        [기존 호환성 유지] 텍스트 프롬프트 생성에 사용될 클래스 목록을 반환합니다.
        기본적으로 파서용 클래스 목록과 동일하게 동작합니다.
        """
        return self.labeled_classes
    
    def get_classes_for_evaluation(self) -> list:
        """
        [추가된 메서드] 모델 성능 평가에 사용될 실제 타겟 클래스 목록을 반환합니다.
        'dummy_label'과 같이 평가에 사용되지 않는 레이블은 제외됩니다.
        """
        # self.classes 리스트에서 'dummy_label'을 필터링하여 반환
        return [cls for cls in self.classes if cls != 'dummy_label']

    def get_target_label_map(self) -> dict:
        """
        [수정] 평가용 클래스 목록을 기준으로 라벨-인덱스 맵을 생성합니다.
        모델의 최종 출력과 매칭시킬 때 사용됩니다.
        """
        # 평가용 클래스 목록을 사용하도록 변경
        return {class_name: i for i, class_name in enumerate(self.get_classes_for_evaluation())}

    def get_class_text_embeddings(self, for_evaluation: bool = False) -> torch.Tensor:
        """
        클래스 이름에 대한 텍스트 임베딩을 생성하여 반환합니다.
        :param for_evaluation: True일 경우, 평가용 클래스 목록을 사용하여 임베딩을 생성합니다.
        """
        if for_evaluation:
            # 평가용 임베딩 생성 및 캐싱
            if self._eval_text_emb is None:
                classes = self.get_classes_for_evaluation()
                prompts = [f"a sound of {c.replace('_', ' ')} in the room" for c in classes]
                model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self._eval_text_emb = model.encode(prompts, convert_to_tensor=True).to(self.device)
            return self._eval_text_emb
        else:
            # 학습용(기존) 임베딩 생성 및 캐싱
            if self._text_emb is None:
                classes = self.get_classes_for_text_prompts()
                prompts = [f"a sound of {c.replace('_', ' ')} in the room" for c in classes]
                model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self._text_emb = model.encode(prompts, convert_to_tensor=True).to(self.device)
            return self._text_emb