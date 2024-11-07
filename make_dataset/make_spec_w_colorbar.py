import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image

# Load the audio file
audio_path = '/chess/project/project1/music/MER_audio_taffc_dataset_wav/renamed/Q4/Q4.MT0000054705.wav'
y, sr = librosa.load(audio_path)  # Load with a sampling rate of 22.05 kHz

# Plot and save the waveform
plt.figure()
librosa.display.waveshow(y, sr=sr, color="blue")
plt.savefig("./sound_libfig.png")
plt.close()

# Compute the STFT
D = librosa.stft(y)  
S, phase = librosa.magphase(D)  # Convert complex values to magnitude and phase
Sdb = librosa.amplitude_to_db(S)  # Convert amplitude to dB scale

# Plot and save the spectrogram in color
plt.figure()
librosa.display.specshow(Sdb, sr=sr)
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
color_image_path = "./spec_img_w_colorbar/color_spec_libfig.png"
plt.savefig(color_image_path)
plt.close()

# Plot and save the spectrogram in grayscale
plt.figure()
librosa.display.specshow(Sdb, sr=sr, cmap='gray')  # Use 'gray' colormap for grayscale
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.savefig("./spec_img_w_colorbar/raw_grayscale_spec_libfig.png")
plt.close()

# Convert the saved color spectrogram to grayscale
color_image = Image.open(color_image_path)
grayscale_image = color_image.convert("L")  # Convert to 1-channel grayscale
grayscale_image.save("./spec_img_w_colorbar/grayscale_spec_libfig.png")
