import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,           # 全体のフォントサイズ
    'axes.titlesize': 12,      # タイトルのフォントサイズ
    'axes.labelsize': 11,      # 軸ラベルのフォントサイズ
    'xtick.labelsize': 11,     # x軸の目盛りラベルのフォントサイズ
    'ytick.labelsize': 11      # y軸の目盛りラベルのフォントサイズ
})

# パラメータ設定
audio_path = '/chess/project/project1/music/MER_audio_taffc_dataset_wav/renamed/Q1/Q1.MT0000040632.wav'
save_dir = '/local/home/matsubara/EfficientNetV2_music_emotion_ctlex/make_dataset/stft_img'
sampling_rate = 22050
window_size = 2048
overlap_rate = 0.5
hop_length = int(window_size * (1 - overlap_rate))
window_function = np.hanning(window_size)
base_color = '#6495ed'

# 保存先ディレクトリ作成
os.makedirs(save_dir, exist_ok=True)

# 音声データを読み込み
y, sr = librosa.load(audio_path, sr=sampling_rate)

# 1. 音楽データ全体の波形
plt.figure(figsize=(12, 4))
plt.plot(y, color='#6495ed')
# plt.title('Full Audio Waveform')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'full_waveform.png'))
plt.close()

# 波形をフレーム化
frames = librosa.util.frame(y, frame_length=window_size, hop_length=hop_length)
frame_count = frames.shape[1]

# 2. 中間の5つのフレームの波形
middle_start = frame_count // 2 - 2
selected_frames = frames[:, middle_start:middle_start + 5]
for i, frame in enumerate(selected_frames.T):
    plt.figure(figsize=(8, 2))
    plt.plot(frame, color= base_color)
    # plt.title(f'Frame {middle_start + i + 1} (Original)')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'frame_{i + 1}_original.png'))
    plt.close()

# 3. 窓関数を掛けた波形
windowed_frames = selected_frames * window_function[:, np.newaxis]
for i, frame in enumerate(windowed_frames.T):
    plt.figure(figsize=(8, 2))
    plt.plot(frame, color=base_color)
    # plt.title(f'Frame {middle_start + i + 1} (Windowed)')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'frame_{i + 1}_windowed.png'))
    plt.close()

# 4. 各フレームの離散フーリエ変換（DFT）の結果
for i, frame in enumerate(windowed_frames.T):
    fft_result = np.abs(np.fft.rfft(frame))
    plt.figure(figsize=(8, 2))
    plt.plot(fft_result, color=base_color)
    # plt.title(f'Frame {middle_start + i + 1} (DFT Result)')
    plt.xlabel('Frequency Bins')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'frame_{i + 1}_dft.png'))
    plt.close()