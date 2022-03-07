import librosa
import os.path
import scipy.io.wavfile


def load(path):
    if os.path.exists(path):
        time_series, sampling_rate = librosa.load(path, sr=None)
        return Audio(path, time_series, sampling_rate)
    else:
        return None


class Audio:
    def __init__(self, path, time_series, sampling_rate):
        folder, filename = os.path.split(path)
        self.folder = folder
        self.path = path
        self.filename = filename
        self.time_series = time_series
        self.sampling_rate = sampling_rate

    @property
    def get_filename(self):
        return self.filename

    @property
    def get_path(self):
        return self.path

    @property
    def get_sampling_rate(self):
        return self.sampling_rate

    def get_id(self):
        return self.filename.split('-')[-1].split('.')[0]

    def get_duration(self):
        return librosa.get_duration(y=self.time_series, sr=self.sampling_rate)

    def __hash__(self):
        return hash(self.time_series)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.time_series == other.time_series
        return NotImplemented

    def save(self, filename):
        scipy.io.wavfile.write(self.folder+"/"+filename, self.sampling_rate, self.time_series)