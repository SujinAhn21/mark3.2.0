# plot_audio.py

import os
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from seed_utils import set_seed
from vild_config import AudioViLDConfig

def create_dummy_audio(file_path, sample_rate=16000, duration=1):
    """테스트용 무작위 오디오 파일 생성"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = np.random.uniform(-0.5, 0.5, sample_rate * duration).astype(np.float32)
    sf.write(file_path, data, sample_rate)
    print(f"[샘플 생성] 임시 테스트 오디오 생성됨: {file_path}")

def plot_waveform(waveform, filename, save_dir, mark_version):
    """Waveform 시각화"""
    try:
        plt.figure(figsize=(10, 3))
        plt.plot(waveform[0].cpu().numpy())
        plt.title("Waveform: " + filename)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        wave_path = os.path.join(save_dir, f"{filename}_waveform_{mark_version}.png")
        plt.savefig(wave_path, dpi=150)
        plt.close()
        print(f"[완료] Waveform plot 저장됨: {wave_path}")
        return wave_path
    except Exception as e:
        print(f"[Error] Waveform 시각화 실패: {e}")
        return "Waveform plot failed"

def plot_mel_spectrogram(waveform, sr, filename, save_dir, config, mark_version):
    """Mel spectrogram 시각화"""
    try:
        if sr != config.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=config.sample_rate)(waveform)
            sr = config.sample_rate

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=config.fft_size,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        )
        mel = mel_transform(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        plt.figure(figsize=(10, 4))
        plt.imshow(mel_db.squeeze(0).cpu().numpy(), origin='lower', aspect='auto', cmap='viridis')
        plt.title("Mel Spectrogram: " + filename)
        plt.xlabel("Time")
        plt.ylabel("Mel bins")
        mel_path = os.path.join(save_dir, f"{filename}_mel_{mark_version}.png")
        plt.savefig(mel_path, dpi=150)
        plt.close()
        print(f"[완료] Mel spectrogram plot 저장됨: {mel_path}")
        return mel_path
    except Exception as e:
        print(f"[Error] Mel spectrogram 시각화 실패: {e}")
        return "Mel plot failed"

def plot_waveform_and_mel(path, save_dir=None, seed_value=42, mark_version="mark3.2.0"):
    set_seed(seed_value)

    # 저장 경로 설정
    if save_dir is None:
        save_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)

    # torchaudio 백엔드 설정 (경고 무시)
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass

    # 오디오 파일 로딩
    try:
        waveform_np, sr = sf.read(path, dtype='float32')
        waveform = torch.from_numpy(waveform_np)
        if waveform.ndim == 2:
            waveform = torch.mean(waveform, dim=1)  # 다채널 평균
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
    except Exception as e:
        print(f"[Error] 오디오 파일 로딩 실패: {path}\n{e}")
        return

    filename = os.path.splitext(os.path.basename(path))[0]
    config = AudioViLDConfig(mark_version=mark_version)

    wave_path = plot_waveform(waveform, filename, save_dir, mark_version)
    mel_path = plot_mel_spectrogram(waveform, sr, filename, save_dir, config, mark_version)

    print(f"[시도 완료] 저장된 시각화 결과 -> Wave: {wave_path}, Mel: {mel_path}")


if __name__ == "__main__":
    test_audio_file = os.path.join(BASE_DIR, "data_wav", "daily_human_talking_50.wav") 
    if not os.path.exists(test_audio_file):
        print(f"[경고] 테스트 오디오 파일 없음: {test_audio_file}")
        create_dummy_audio(test_audio_file)

    plot_waveform_and_mel(test_audio_file, seed_value=42)
    