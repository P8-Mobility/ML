import os
import librosa
import audio
import transformer
from audio import Audio
import soundfile as sf


class DataLoader:
    def __init__(self, audio_list=None):
        if audio_list is None:
            audio_list = []

        self.__data = audio_list
        self.__duration_scale = 0
        self.__duration_sum = 0

    def clear(self):
        self.__data.clear()
        self.__duration_scale = 0
        self.__duration_sum = 0

    def add_folder_to_model(self, path):
        if os.path.isdir(path):
            for filename in os.listdir(path):
                self.add_file_to_model(path + "/" + filename)

    def add_file_to_model(self, path):
        if os.path.isfile(path):
            self.__data.append(audio.load(path))

    def load_file(self, path: str):
        if os.path.isfile(path):
            audio_file = audio.load(path)
            self.preprocessing(audio_file, False)
            self.scale(audio_file)
            return audio_file
        return None

    def fit(self, with_mfccs: bool):
        for audio_file in self.__data:
            self.preprocessing(audio_file, with_mfccs)
            self.__duration_sum += audio_file.get_original_duration()

        self.__duration_scale = self.__duration_sum / len(self.__data)

        for audio_file in self.__data:
            self.scale(audio_file)

    def size(self):
        return len(self.__data)

    def get_data_files(self):
        return self.__data

    # def get_dataframe(self):
    #     file_names = []
    #     time_series = {}
    #
    #     for audio_file in self.__data:
    #         file_names.append(audio_file.get_filename)
    #         for i in range(len(audio_file.time_series)):
    #             time_series[i] = audio_file.time_series[i]
    #
    #     pdo = pd.DataFrame({"filename": file_names})
    #     return pdo.append(time_series, ignore_index=True)

    def preprocessing(self, audio_file: Audio, with_mfccs: bool):
        audio_file.time_series = librosa.to_mono(audio_file.time_series)
        transformer.normalize(audio_file)
        transformer.remove_noice(audio_file)
        transformer.trim(audio_file, 20)
        if with_mfccs:
            transformer.mfccs(audio_file)
        #sf.write(audio_file.get_filename, audio_file.time_series, audio_file.get_sampling_rate)

    def scale(self, audio_file: Audio):
        audio_file.time_series = librosa.effects.time_stretch(audio_file.get_orignial_time_series(), rate=audio_file.get_original_duration() / self.__duration_scale)


