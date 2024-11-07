import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# 音声ファイルのパス
audio_path = '/chess/project/project1/music/MER_audio_taffc_dataset_wav/renamed/Q1/Q1.MT0000040632.wav'

# 保存先ディレクトリ
save_dir = './window_waveform'

# ディレクトリが存在しない場合は作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# サンプリング周波数とオーバーラップ率の設定
sampling_rate = 22050
overlap_rate = 0.5

# 音声データを読み込み（指定したサンプリング周波数でリサンプリング）
y, sr = librosa.load(audio_path, sr=sampling_rate)

# ウィンドウサイズを指定
window_sizes = [256, 2048]

# 各ウィンドウサイズで10フレームだけ波形を切り出して保存
for window_size in window_sizes:
    # オーバーラップに応じた hop_length の計算
    hop_length = int(window_size * (1 - overlap_rate))

    # 波形を指定したウィンドウサイズで切り出し
    frames = librosa.util.frame(y, frame_length=window_size, hop_length=hop_length)
    
    # 10フレームだけ取得してプロット・保存
    for i, frame in enumerate(frames.T[:10]):  # 10フレームのみ取得
        plt.figure(figsize=(10, 4))
        plt.plot(frame)
        plt.title(f'Waveform Window Size {window_size} - Frame {i} (Sampling {sampling_rate}, Overlap {overlap_rate})')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        
        # ファイル名を生成
        filename = f'waveform_window_{window_size}_frame_{i}.png'
        filepath = os.path.join(save_dir, filename)
        
        # 図を保存
        plt.savefig(filepath)
        plt.close()

    print(f'Window size {window_size}: First 10 frames saved to {save_dir}')
