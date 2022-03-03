from pydub import AudioSegment
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
    duration_list = []
    path = "sound/2ndBatch"

    for file in os.listdir(path):
        time_series, sampling_rate = librosa.load(path + "/" + file)  # Makes floating point time series

        # Normalize (make all batches the same level)
        max_peak = np.max(np.abs(time_series))
        ratio = 1 / max_peak
        time_series = time_series * ratio

        # Reduce noice and trim
        time_series = nr.reduce_noise(y=time_series, sr=sampling_rate)
        time_series, index = librosa.effects.trim(time_series, top_db=30)

        # Finds duration for avg
        duration = librosa.get_duration(y=time_series, sr=sampling_rate)
        duration_list.append(duration)
        files.append(time_series)

    avg_duration = sum(duration_list) / len(duration_list)

    for index in range(len(files)):
        # stretches duration so all files is avg length
        duration = librosa.get_duration(y=file, sr=sampling_rate)
        files[index] = librosa.effects.time_stretch(file, rate=duration / avg_duration)
        sf.write(str(file) + '.wav', file, sampling_rate, subtype='PCM_24')
    return files


if __name__ == '__main__':
    sound_files = preprocess_sound()
    #plot_specgram_plt('sound/1stBatch/pære_1.m4a', "m4a", 1)
    #plot_specgram_plt('sound/1stBatch/bære_1.m4a', "m4a", 2)

    #plot_specgram_librosa('sound/1stBatch/pære_1.m4a', "m4a", 3)
    #plot_specgram_librosa('sound/1stBatch/bære_1.m4a', "m4a", 4)

    plt.show()
