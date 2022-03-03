from pydub import AudioSegment, effects
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import numpy as np
import librosa
import librosa.display
import os
import soundfile as sf
import noisereduce as nr


def convert_audio(audio):
    wav_file = mktemp('.wav')
    audio.export(wav_file, format="wav")  # convert to wav
    return wav_file


def plot_specgram_plt(filename, fileformat, no):
    audio = AudioSegment.from_file(filename, format=fileformat)
    converted_audio = convert_audio(audio)
    sampling_rate, data = wavfile.read(converted_audio)

    plt.figure(no)
    plt.specgram(data, Fs=sampling_rate, NFFT=128, noverlap=0)


def plot_specgram_librosa(filename, fileformat, no):
    # Documentation: https://librosa.org/doc
    audio = AudioSegment.from_file(filename, format=fileformat)
    converted_audio = convert_audio(audio)
    time_series, sampling_rate = librosa.load(converted_audio, sr=None)  # Makes floating point time series
    time_series_trimmed, index = librosa.effects.trim(time_series, top_db=30)

    window_size = 1000
    hop_length = 128  # Default
    window = np.hanning(window_size)  # Returns hanning window
    stft = librosa.core.spectrum.stft(time_series_trimmed, n_fft=window_size, hop_length=hop_length, window=window)  # STFT: Short-time Fourier transform
    out = 2 * np.abs(stft) / np.sum(window)  # Finds amplitude?

    plt.figure(no)
    librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), y_axis='log', x_axis='time', sr=sampling_rate)


def preprocess_sound():
    files = []
    dir_list = os.listdir()
    path = "sound/2ndBatch"
    for file in os.listdir(path):
        time_series, sampling_rate = librosa.load(path + "/" + file)  # Makes floating point time series

        # Normalize (make all batches the same level)
        max_peak = np.max(np.abs(time_series))
        ratio = 1 / max_peak
        time_series = time_series * ratio

        # Reduce noice and trim
        reduced_noise = nr.reduce_noise(y=time_series, sr=sampling_rate)
        time_series_trimmed, index = librosa.effects.trim(reduced_noise, top_db=30)

        files.append(time_series_trimmed)
        #sf.write(str(time_series_trimmed) + '.wav', time_series_trimmed, sampling_rate, subtype='PCM_24')
    return files


if __name__ == '__main__':
    sound_files = preprocess_sound()
    #plot_specgram_plt('sound/1stBatch/pære_1.m4a', "m4a", 1)
    #plot_specgram_plt('sound/1stBatch/bære_1.m4a', "m4a", 2)

    #plot_specgram_librosa('sound/1stBatch/pære_1.m4a', "m4a", 3)
    #plot_specgram_librosa('sound/1stBatch/bære_1.m4a', "m4a", 4)

    plt.show()
