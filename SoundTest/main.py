from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import numpy as np
import librosa
import librosa.display


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

    window_size = 1000
    hop_length = 128  # Default
    window = np.hanning(window_size)  # Returns hanning window
    stft = librosa.core.spectrum.stft(time_series, n_fft=window_size, hop_length=hop_length, window=window)  # STFT: Short-time Fourier transform
    out = 2 * np.abs(stft) / np.sum(window)  # Finds amplitude?

    plt.figure(no)
    librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), y_axis='log', x_axis='time', sr=sampling_rate)


if __name__ == '__main__':
    plot_specgram_plt('pære_1.m4a', "m4a", 1)
    plot_specgram_plt('bære_1.m4a', "m4a", 2)

    plot_specgram_librosa('pære_1.m4a', "m4a", 3)
    plot_specgram_librosa('bære_1.m4a', "m4a", 4)

    plt.show()
