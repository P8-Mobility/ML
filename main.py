import allosaurus.app as allo
import allosaurus.audio
import fine_tune as ft
from pathlib import Path
from processing.data_loader import DataLoader


def main():
    # process()
    ft.fine_tune()
    # recognize()
    return


def recognize():
    model = allo.read_recognizer(alt_model_path=Path('paereModel'))
    files = loader.get_data_files()

    for file in files:
        aud = allosaurus.audio.Audio(
            file.time_series,
            file.get_sampling_rate)
        res: str = model.recognize(aud)
        print(file.get_filename + ": " + res)


def process():
    loader = DataLoader()
    loader.change_setting("scale_length", False)
    loader.add_folder_to_model('data/unprocessed')
    loader.fit()
    loader.store_processed_files()


if __name__ == "__main__":
    main()

# run inference -> æ l u s ɔ ɹ s
# print("pære: " + )
# print("bære: " + model.recognize('bre1.wav'))
# print("avg: " + model.recognize('avg_sound.wav'))
