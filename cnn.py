import numpy
import tensorflow as tf
from matplotlib import pyplot as plt

import SoundTest.main


def run():
    files = SoundTest.main.preprocess_sound()
    files = numpy.asarray([tf.reshape(f, [1, 1, 10240]) for f in files])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(tensor=files[0], dtype=tf.float32),
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
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # print(train_labels)
    labels = numpy.array(numpy.zeros(len(files)))
    history = model.fit(x=files, y=labels, epochs=10)
    print(model(files[0]))
