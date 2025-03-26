import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image
import os
from scipy.ndimage import zoom

def make_gray_spectrogram(audio_path, sampling_rate=22050, window_size=2048, overlap_ratio=0.5, output_dir="./spectrogram_outputs"):
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
    img = librosa.display.specshow(Sdb, sr=sr, cmap='gray', hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(img, label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    # Save the figure
    raw_grayscale_path = f"{output_dir}/raw_grayscale_spectrogram.png"
    plt.savefig(raw_grayscale_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved raw grayscale spectrogram: {raw_grayscale_path}")

def make_color_spectrogram(audio_path, sampling_rate=22050, window_size=2048, overlap_ratio=0.5, output_dir="./spectrogram_outputs"):
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
    
    # Save raw color spectrogram with colorbar and axis ticks
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(Sdb, sr=sr, cmap='coolwarm', hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(img, label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    # Save the figure
    raw_color_path = f"{output_dir}/raw_color_spectrogram.png"
    plt.savefig(raw_color_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved raw color spectrogram: {raw_color_path}")

# Example usage
audio_path = '/chess/project/project1/music/MER_audio_taffc_dataset_wav/Q1/MT0000040632.wav'
make_gray_spectrogram(audio_path)
make_color_spectrogram(audio_path)

def make_gray_spectrogram_with_correct_axes(audio_path, sampling_rate=22050, window_size=2048, overlap_ratio=0.5, output_dir="./spectrogram_outputs"):
    """
    Generate and save spectrogram images with original axis values preserved.
    
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
    
    # Get original time and frequency axis values
    time_axis = librosa.frames_to_time(range(Sdb.shape[1]), sr=sr, hop_length=hop_length)
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=window_size)
    
    # Resize spectrogram
    desired_size = (384, 384)  # Target size
    resized_Sdb = zoom(Sdb, (desired_size[0] / Sdb.shape[0], desired_size[1] / Sdb.shape[1]))
    
    # Plot resized spectrogram while keeping the original axis values
    plt.figure(figsize=(7.43,6))
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]  # Keep original axis values
    plt.imshow(resized_Sdb, aspect='auto', cmap='gray', extent=extent, origin='lower')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    resized_grayscale_path = f"{output_dir}/resized_grayscale_spectrogram_384x384_with_correct_axes.png"
    plt.savefig(resized_grayscale_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved resized grayscale spectrogram with correct axes: {resized_grayscale_path}")

make_gray_spectrogram_with_correct_axes(audio_path)

def make_color_spectrogram_with_correct_axes(audio_path, sampling_rate=22050, window_size=2048, overlap_ratio=0.5, output_dir="./spectrogram_outputs"):
    """
    Generate and save spectrogram images with original axis values preserved.
    
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
    
    # Get original time and frequency axis values
    time_axis = librosa.frames_to_time(range(Sdb.shape[1]), sr=sr, hop_length=hop_length)
    freq_axis = librosa.fft_frequencies(sr=sr, n_fft=window_size)
    
    # Resize spectrogram
    desired_size = (384, 384)  # Target size
    resized_Sdb = zoom(Sdb, (desired_size[0] / Sdb.shape[0], desired_size[1] / Sdb.shape[1]))
    
    # Plot resized spectrogram while keeping the original axis values
    plt.figure(figsize=(7.43,6))
    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]  # Keep original axis values
    plt.imshow(resized_Sdb, aspect='auto', cmap='coolwarm', extent=extent, origin='lower')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    resized_color_path = f"{output_dir}/resized_color_spectrogram_384x384_with_correct_axes.png"
    plt.savefig(resized_color_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved resized grayscale spectrogram with correct axes: {resized_color_path}")

make_color_spectrogram_with_correct_axes(audio_path)