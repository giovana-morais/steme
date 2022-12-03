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


def evaluate_model(ballroom, evaluation_file, model, kmin, kmax):
    print("Evaluating model")
    if not os.path.isfile(evaluation_file):
        print(f"Generating {evaluation_file}")
        with h5py.File(evaluation_file, "a") as whf:
            for track_id in ballroom.track_ids:
                fixed_shift_predictions = []
                random_shift_predictions = []

                theta = dt.variables_non_linear()
                x, sr = ballroom.track(track_id).audio
                T, t, freqs = audio.tempogram(
                    x, sr, window_size_seconds=10, t_type="hybrid", theta=theta)
                reference_tempo = ballroom.track(track_id).tempo

                for i in range(T.shape[1]):
                    s1, sh1, s2, sh2, _ = dt.get_tempogram_slices(
                        T, slice_idx=i, kmin=kmin, kmax=kmax)

                    fixed_s1, fixed_sh1, _, _, _ = dt.get_tempogram_slices(
                        T, slice_idx=i, kmin=kmin, kmax=kmax, shift_1=0, shift_2=0)

                    s1 = s1[np.newaxis, :]
                    fixed_s1 = fixed_s1[np.newaxis, :]

                    xhat1, xhat2, y1, y2 = model.predict(
                        [s1, s1, sh1, sh1], verbose=0)
                    fixed_xhat1, fixed_xhat2, fixed_y1, fixed_y2 = model.predict(
                        [fixed_s1, fixed_s1, fixed_sh1, fixed_sh1], verbose=0)

                    random_shift_predictions.append(y1[0][0])
                    fixed_shift_predictions.append(fixed_y1[0][0])

                # predicted_tempo_linear = np.array(predictions)*a+b
                # predicted_tempo_quadratic = quad(np.array(predictions))
                baseline_tempo = np.take(freqs, np.argmax(T, axis=-2))

                g = whf.create_group(track_id)
                g["reference_tempo"] = reference_tempo
                g["fixed_shift_model_output"] = fixed_shift_predictions
                # g["random_shift_model_output"] = random_shift_predictions
                # g["predicted_tempo_linear"] = predicted_tempo_linear
                # g["predicted_tempo_quadratic"] = predicted_tempo_quadratic
                g["baseline_tempo"] = baseline_tempo
                g["T"] = T.copy()
                g["t"] = t
                g["freqs"] = freqs
    else:
        print(f"{evaluation_file} already exists")

    return


def main(
        model_name,
        dataset_name,
        dataset_type,
        synthetic,
        n_predictions,
        kmin,
        kmax,
        tmin,
        n_bins,
        bins_per_octave,
        **kwargs):

    theta = dt.variables_non_linear(tmin, bins_per_octave, n_bins)

    response = dt.read_dataset_info(dataset_name)

    main_file = response["main_file"]
    train_file = response["train_file"]
    validation_file = response["validation_file"]
    main_filepath = response["main_filepath"]
    train_filepath = response["train_filepath"]
    validation_filepath = response["validation_filepath"]

    tmin = response["tmin"]
    tmax = response["tmax"]
    distribution = response["distribution"]

    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = tf.keras.models.load_model(model_path)

    # evaluate on ballroom
    ballroom = loader.custom_dataset_loader(
        path=DATASET_FOLDER, dataset_name="ballroom", folder="")
    evaluation_file = os.path.join(DATA_FOLDER, f"{model_name}_evaluation.h5")
    main_file = os.path.join(DATA_FOLDER, "ballroom.h5")
    evaluate_model(ballroom, evaluation_file, model, kmin, kmax)

    return


if __name__ == "__main__":
    import fire
    fire.Fire(main)
