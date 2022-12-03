import os
import types
import loader
import librosa
import numpy as np
from mirdata import annotations
from mirdata.core import cached_property
from typing import BinaryIO, Optional, TextIO, Tuple

MAX_STR_LEN = 100

def custom_loader(path, dataset_name, folder='datasets'):
    print(f'Loading {dataset_name} through custom loader')
    datasetdir = os.path.join(path, folder, dataset_name)
    dataset = loader.Dataset(data_home=os.path.join(datasetdir, 'audio'),
                             annotations_home=os.path.join(datasetdir, 'annotations'))
    tracks = dataset.load_tracks()
    return tracks

def custom_dataset_loader(path, dataset_name, folder='datasets'):
    print(f'Loading {dataset_name} through custom loader')
    datasetdir = os.path.join(path, folder, dataset_name)
    dataset = loader.Dataset(data_home=os.path.join(datasetdir, 'audio'),
                             annotations_home=os.path.join(datasetdir, 'annotations'))
    return dataset

class Track:
    def __init__(self, track_id, dataset_name, index):
        self.track_id = track_id
        self._dataset_name = dataset_name
        self._track_paths = index[track_id]

        self.audio_path = self.get_path("audio")
        self.beats_path = self.get_path("beats")
        self.tempo_path = self.get_path("tempo")

    @cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @cached_property
    def tempo(self) -> Optional[float]:
        return load_tempo(self.tempo_path)

    @cached_property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        return load_audio(self.audio_path)

    def get_path(self, key):
        if self._track_paths[key] is None:
            return None
        else:
            return self._track_paths[key]

    def __repr__(self):
        properties = [v for v in dir(self.__class__) if not v.startswith("_")]
        attributes = [
            v for v in dir(self) if not v.startswith("_") and v not in properties
        ]

        repr_str = "Track(\n"

        for attr in attributes:
            val = getattr(self, attr)
            if isinstance(val, str):
                if len(val) > MAX_STR_LEN:
                    val = "...{}".format(val[-MAX_STR_LEN:])
                val = '"{}"'.format(val)
            repr_str += "  {}={},\n".format(attr, val)

        for prop in properties:
            val = getattr(self.__class__, prop)
            if isinstance(val, types.FunctionType):
                continue

            if val.__doc__ is None:
                doc = ""
            else:
                doc = val.__doc__

            val_type_str = doc.split(":")[0]
            repr_str += "  {}: {},\n".format(prop, val_type_str)

        repr_str += ")"
        return repr_str

def load_beats(fhandle: TextIO) -> annotations.BeatData:
    beats = np.loadtxt(fhandle)
    times = beats[:, 0]
    positions = beats[:, 1]
    beat_data = annotations.BeatData(
        times=times, time_unit="s", positions=positions, position_unit="bar_index"
    )
    return beat_data

def load_tempo(fhandle: TextIO) -> float:
    tempo = np.loadtxt(fhandle)
    return float(tempo)

def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    audio, sr = librosa.load(fhandle, sr=44100, mono=True)
    return audio, sr

def brid_indexing_function(filename):
    return filename[1:5]

def candombe_indexing_function(filename):
    return os.path.splitext(filename)[0]

def indexing_function(filename):
    return os.path.splitext(filename)[0]

class Dataset:
    def __init__(self, data_home='./datasets/brid/audio', annotations_home='./datasets/brid/', dataset_name='brid', indexing_function=indexing_function):
        self.dataset_name = dataset_name
        self.data_home = data_home
        self.annotations_home = annotations_home

        self._index = {}
        beats_home = os.path.join(self.annotations_home, 'beats')
        tempo_home = os.path.join(self.annotations_home, 'tempo')
        for root, dirs, files in os.walk(self.data_home):
            for name in files:
                if not name=='.DS_Store':
                    aux_dict = {'audio': os.path.join(root, name),
                                'beats': os.path.join(beats_home, name.replace('.wav', '.beats')),
                                'tempo': os.path.join(tempo_home, name.replace('.wav', '.bpm'))}
                    file_code = indexing_function(name)
                    self._index[file_code] = aux_dict

    def track(self, track_id):
        return Track(track_id, self.dataset_name, self._index)

    def load_tracks(self):
        return {track_id: self.track(track_id) for track_id in self.track_ids}

    @property
    def track_ids(self):
        return list(self._index.keys())

    @property
    def name(self):
        return self.dataset_name
