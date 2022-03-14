import allosaurus.allosaurus.app as allo
import allosaurus.allosaurus.audio
import data_loader
import fine_tune as ft
from pathlib import Path


def main():
    ft.fine_tune()
    recognize()
    return


def recognize():
    model = allo.read_recognizer(alt_model_path=Path('paereModel'))

    loader = data_loader.DataLoader()
    loader.add_folder_to_model('files/')
    loader.fit(False)
    files = loader.get_data_files()

    for file in files:
        aud = allosaurus.allosaurus.audio.Audio(
            file.time_series,
            file.get_sampling_rate)
        res: str = model.recognize(aud)
        print(file.get_filename + ": " + res)


if __name__ == "__main__":
    main()

# run inference -> æ l u s ɔ ɹ s
# print("pære: " + )
# print("bære: " + model.recognize('bre1.wav'))
# print("avg: " + model.recognize('avg_sound.wav'))
