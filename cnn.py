import librosa
import numpy
import numpy as np
import tensorflow as tf
from keras.layers import Reshape
from numpy.ma import indices


def run():
    # Documentation: https://librosa.org/doc
    files_paths = librosa.util.find_files("files/", ext=['wav'])
    files = np.empty([6, 501, 809])
    for file_path in files_paths:
        time_series, sampling_rate = librosa.load(file_path, sr=48000)  # Makes floating point time series
        window_size = 1000
        hop_length = 128  # Default
        window = np.hanning(window_size)  # Returns hanning window
        stft = librosa.amplitude_to_db(
            np.abs(librosa.core.spectrum.stft(time_series, n_fft=window_size, hop_length=hop_length, window=window)),
            ref=np.max)  # STFT: Short-time Fourier transform
        out: tf.Tensor = 2 * np.abs(stft) / np.sum(window)  # Finds amplitude?
        out = tf.reshape(out, (500, 800))
        files.__add__(out)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=files[0]),
        tf.keras.layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model(files[0])
