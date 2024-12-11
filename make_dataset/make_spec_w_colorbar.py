import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image
import os
from scipy.ndimage import zoom

def make_spectrogram(audio_path, sampling_rate=22050, window_size=2048, overlap_ratio=0.5, output_dir="./spectrogram_outputs"):
    """
    Generate and save spectrogram images.
    
    Parameters:
    - audio_path: Path to the audio file.
    - sampling_rate: Sampling rate for loading the audio.
    - window_size: Window size for STFT.
    - overlap_ratio: Overlap ratio for STFT.
    - output_dir: Directory to save the output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sampling_rate)
    
    # Calculate hop length based on overlap ratio
    hop_length = int(window_size * (1 - overlap_ratio))
    
    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(y, n_fft=window_size, hop_length=hop_length)
    S, _ = librosa.magphase(D)  # Get magnitude and phase
    Sdb = librosa.amplitude_to_db(S)  # Convert magnitude to dB scale
    # スペクトログラムを 384x384 にリサイズ
    desired_size = (384, 384)  # 目標サイズ
    resized_Sdb = zoom(Sdb, (desired_size[0] / Sdb.shape[0], desired_size[1] / Sdb.shape[1]))
    
    # Save raw grayscale spectrogram with colorbar and axis ticks
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(Sdb, sr=sr, cmap='gray', hop_length=hop_length)
    plt.colorbar(img, label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    # Save the figure
    raw_grayscale_path = f"{output_dir}/raw_grayscale_spectrogram.png"
    plt.savefig(raw_grayscale_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    
    # Resize the grayscale spectrogram to 384x384 and save
    plt.figure(figsize=(6, 6))
    librosa.display.specshow(resized_Sdb, sr=sr, cmap='gray', hop_length=hop_length)
    # plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    resized_grayscale_path = f"{output_dir}/resized_grayscale_spectrogram_384x384.png"
    plt.savefig(resized_grayscale_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved raw grayscale spectrogram: {raw_grayscale_path}")
    print(f"Saved resized grayscale spectrogram: {resized_grayscale_path}")

# Example usage
audio_path = '/chess/project/project1/music/MER_audio_taffc_dataset_wav/renamed/Q4/Q4.MT0000054705.wav'
make_spectrogram(audio_path)
