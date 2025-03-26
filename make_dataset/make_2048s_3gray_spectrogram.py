import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import re
from torchvision import transforms
from torchaudio.transforms import TimeMasking, FrequencyMasking

dataset_path = "/chess/project/project1/music/MER_audio_taffc_dataset_wav/"
renamed_dataset_path = dataset_path + "/renamed/"
grayscale_spec_dataset_path = dataset_path + "/spec/3grayscale"

# グレースケール用ディレクトリ作成
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    os.makedirs(f"{grayscale_spec_dataset_path}/{q}/2048s", exist_ok=True)

# pyplot の詳細設定
plt.rcParams["figure.figsize"] = [3.84, 3.84]    # 図の縦横のサイズ([横(inch),縦(inch)])
plt.rcParams["figure.dpi"] = 100                # dpi (dots per inch)
plt.rcParams["figure.subplot.left"] = 0         # 余白なし
plt.rcParams["figure.subplot.bottom"] = 0
plt.rcParams["figure.subplot.right"] = 1
plt.rcParams["figure.subplot.top"] = 1

# audio_list の取得
def path_to_audiofiles(dir_folder):
    list_of_audio = []
    for file in os.listdir(dir_folder):
        if file.endswith(".wav"):
            directory = "%s/%s" % (dir_folder, file)
            list_of_audio.append(directory)
    return list_of_audio

# スペクトログラム画像生成（グレースケール）
def make_spectrogram(audio_path, sampling, window, overlap):
    sample_freq, signal = librosa.load(audio_path, sr=sampling)  # サンプリング周波数 sampling kHzで読み込み
    sftp = librosa.stft(sample_freq, n_fft=window, hop_length=overlap)  # STFT
    strength, phase = librosa.magphase(sftp)  # 複素数を強度と位相へ変換
    db = librosa.amplitude_to_db(strength)  # 強度をdb単位へ変換
    return db, signal

def save_spectrogram(save_path, db, signal):
    plt.figure()
    librosa.display.specshow(db, sr=signal, cmap="gray")  # グレースケールのスペクトログラムを表示
    plt.savefig(save_path)
    plt.close()

# サンプリング周波数・窓サイズ・オーバラップ率の設定
sampling = 22050
window_values = [2046, 2048, 2050]
overlap_rate = 0.5

# 各カテゴリでスペクトログラム生成
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    filelist = path_to_audiofiles(f"{renamed_dataset_path}/{q}")
    for file in filelist:
        filename = os.path.basename(file)
        filename_no_extension = os.path.splitext(filename)[0]
        raw_path = f"{grayscale_spec_dataset_path}/{q}/2048s/{filename_no_extension}"
        for window_value in window_values:
            # パラメータ設定
            window = window_value
            overlap = int(window * overlap_rate)
            db, signal = make_spectrogram(file, sampling, window, overlap)
            # 保存するファイル名設定
            param_data = f"_{sampling}_{window}_{int(overlap_rate*100)}"
            save_name = raw_path + param_data + ".png"
            # ファイルが存在しない場合のみ実行
            if not os.path.exists(save_name):
                save_spectrogram(save_name, db, signal)
                print(f"[LOG] {filename} -> /spec/3grayscale/{q}/2048s/{filename_no_extension}{param_data}.png")
    print("--------------------")

# ファイル数カウント関数
def count_file(folder_path):
    import pathlib
    return sum(1 for path in pathlib.Path(folder_path).iterdir() if path.is_file())

# 各ディレクトリのデータ数を表示
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    count = count_file(f"{grayscale_spec_dataset_path}/{q}/2048s")
    print(f"[INFO] Datas in spec/3grayscale/{q}/2048s: {count}")
