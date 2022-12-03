# -*- coding: utf-8 -*-
import os
import random

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

import audio
import dataset as dt
import loader
import metrics
import utils
from paths import *

def generate_predictions(mirdata_dataset, mirdata_dataset_data_folder,
        model_name, kmin, kmax, track_file):

    model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, model_name))

    predictions_folder = os.path.join(DATA_FOLDER, f"predictions/{model_name}")
    print(f"predictions_folder {predictions_folder}")

    predictions_folder = os.path.join(DATA_FOLDER, f"predictions/{model_name}_fixed_shift")

    if not os.path.isdir(predictions_folder):
        os.mkdir(predictions_folder)

    with h5py.File(track_file, "r") as hf:
        total = len(hf.keys())

        for idx, track_id in enumerate(hf.keys()):
            print(f"processing {track_id}. {idx}/{total}")
            dest_path = os.path.join(predictions_folder, track_id)
            predictions = []

            if os.path.isfile(f"{dest_path}.npz"):
                print(f"{dest_path} exists")
                continue

            track_data = np.load(os.path.join(mirdata_dataset_data_folder, f"{track_id}.npz"))
            T = track_data["T"]
            t = track_data["t"]
            freqs = track_data["freqs"]

            for i in range(T.shape[1]):
                s1, sh1, _, _, _ = dt.get_tempogram_slices(T, slice_idx=i, kmin=kmin, kmax=kmax, shift_1=0,
                shift_2=0)

                s1 = s1[np.newaxis, :]

                xhat1, xhat2, y1, y2 = model.predict([s1, s1, sh1, sh1], verbose=0)

                predictions.append(y1[0][0])

            baseline_tempo = np.take(freqs, np.argmax(T, axis=-2))

            np.savez(
                dest_path,
                baseline_tempo=baseline_tempo,
                prediction=np.array(predictions)
            )

    return

def main(model_name, dataset_name, kmin, kmax, tmin, n_bins, bins_per_octave, **kwargs):

    theta = dt.variables_non_linear(tmin, bins_per_octave, n_bins)

    response = dt.read_dataset_info(dataset_name)

    main_file = response["main_file"]
    train_file = response["train_file"]
    validation_file = response["validation_file"]
    main_filepath = response["main_filepath"]
    train_filepath = response["train_filepath"]
    validation_filepath = response["validation_filepath"]

    distribution = response["distribution"]

    # generate predictions
    gtzan, _, _ = dt.gtzan_data()
    gtzan_data_folder = os.path.join(PRE_COMPUTED_DATA_FOLDER, f"gtzan_{tmin}_{n_bins}_{bins_per_octave}_fourier")
    # ballroom = loader.custom_dataset_loader(path=DATASET_FOLDER, dataset_name="ballroom", folder="")
    # ballroom_data_folder = os.path.join(PRE_COMPUTED_DATA_FOLDER, f"ballroom_{tmin}_{n_bins}_{bins_per_octave}")
    print(f"ballroom_data_folder: {gtzan_data_folder}")
    generate_predictions(gtzan, gtzan_data_folder, model_name, kmin, kmax, validation_filepath)
    return

if __name__ == "__main__":
    import fire
    fire.Fire(main)
