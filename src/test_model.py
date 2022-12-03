# -*- coding: utf-8 -*-
import datetime
import os
import random

import h5py
import keras
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import tensorflow as tf

import audio
import dataset as dt
import loader
import metrics
import models
import utils
from paths import *


def calibrate_model(
        model,
        kmin,
        kmax,
        mirdata_dataset,
        n_predictions=100,
        shift=False):
    tempi = [mirdata_dataset.track(i).tempo for i in mirdata_dataset.track_ids]
    track = [i for i in mirdata_dataset.track_ids]
    tracks = []
    bpm_tracks = []

    if mirdata_dataset.name == "gtzan_genre":
        tempi.remove(None)
        track.remove("reggae.00086")

    tempi = np.array(tempi)
    track = np.array(track)

    ordered_indexes = np.argsort(tempi)
    tempi = tempi[ordered_indexes]
    track = track[ordered_indexes]

    track_dict = {i: j for i, j in zip(tempi, track)}

    intervals = np.arange(30, 350, 10)
    bpm_dict = {}

    theta = dt.variables_non_linear()

    # TODO: instead of calculating tempograms everytime, we could only load the
    # file from the dataset name
    print("Sampling tracks for calibration")
    for idx in range(len(intervals) - 1):
        try:
            interval = tempi[(tempi > intervals[idx]) &
                             (tempi < intervals[idx + 1])]
            bpm = random.choice(interval)
            bpm_dict[bpm] = {}
            x, sr = mirdata_dataset.track(track_dict[bpm]).audio
            T, t, bpms = audio.tempogram(
                x, sr, window_size_seconds=10, t_type="hybrid", theta=theta)

            bpm_dict[bpm]["audio"] = x
            bpm_dict[bpm]["T"] = T
            bpm_dict[bpm]["t"] = t
            bpm_dict[bpm]["freqs"] = bpms

            tracks.append(track_dict[bpm])
            bpm_tracks.append(bpm)
            print(f"[{intervals[idx]}, {intervals[idx+1]}] - {bpm}")
        except IndexError:
            print(
                f"no index for interval [{intervals[idx]}, {intervals[idx+1]}]")

    print("Calibrating model")
    model_output = np.zeros(len(bpm_dict.keys()))
    j = 0
    for bpm in bpm_dict.keys():
        T = bpm_dict[bpm]["T"]

        preds = np.zeros(n_predictions)

        for i in range(n_predictions):
            s1, sh1, s2, sh2, _ = dt.get_tempogram_slices(
                T, kmin=kmin, kmax=kmax)
            s1 = s1[np.newaxis, :]
            s2 = s1[np.newaxis, :]

            if not shift:
                s2 = s1
                sh2 = sh1

            xhat1, xhat2, y1, y2 = model.predict([s1, s2, sh1, sh2], verbose=0)
            preds[i] = y1[0][0]

        model_output[j] = np.median(np.array(preds))
        j += 1

    quad = np.poly1d(np.polyfit(model_output, list(bpm_dict.keys()), 2))
    a, b = utils.get_slope(model_output, list(bpm_dict.keys()))

    return a, b, quad


def evaluate_model(ballroom, evaluation_file, model, a, b, quad):
    with h5py.File(evaluation_file, "a") as whf:
        for track_id in ballroom.track_ids:
            predictions = []

            theta = dt.variables_non_linear()
            x, sr = ballroom.track(track_id).audio
            T, t, freqs = audio.tempogram(
                x, sr, window_size_seconds=10, t_type="hybrid", theta=theta)
            reference_tempo = ballroom.track(track_id).tempo

            for i in range(T.shape[1]):
                s1, sh1, s2, sh2, _ = dt.get_tempogram_slices(
                    T, slice_idx=i, kmin=11, kmax=19)
                # range between 0,1
                s1 /= s1.max()
                s2 /= s2.max()

                s1 = s1[np.newaxis, :]

                xhat1, xhat2, y1, y2 = model.predict(
                    [s1, s1, sh1, sh1], verbose=0)
                predictions.append(y1[0][0])

            predicted_tempo_linear = np.array(predictions) * a + b
            predicted_tempo_quadratic = quad(np.array(predictions))
            baseline_tempo = np.take(freqs, np.argmax(T, axis=-2))

            g = whf.create_group(track_id)
            g["reference_tempo"] = reference_tempo
            g["predicted_tempo_linear"] = predicted_tempo_linear
            g["predicted_tempo_quadratic"] = predicted_tempo_quadratic
            g["baseline_tempo"] = baseline_tempo
            g["T"] = T.copy()
            g["t"] = t
            g["freqs"] = freqs

    return


def get_metrics(evaluation_file):
    baseline_metrics = {}
    predicted_metrics = {}

    with h5py.File(evaluation_file, "r") as hf:
        for key, value in hf.items():
            baseline_tempo = value["baseline_tempo"][:]
            reference_tempo = value["reference_tempo"][()]
            predicted_tempo_linear = value["predicted_tempo_linear"][:]
            T = value["T"][:]
            # t = value["t"][:]
            # freqs = value["freqs"][:]

            baseline_acc1 = metrics.acc1(
                reference_tempo, np.median(baseline_tempo))
            baseline_acc2 = metrics.acc2(
                reference_tempo, np.median(baseline_tempo))

            predicted_acc1 = metrics.acc1(
                reference_tempo, np.median(predicted_tempo_linear))
            predicted_acc2 = metrics.acc2(
                reference_tempo, np.median(predicted_tempo_linear))

            baseline_metrics[key] = {}
            baseline_metrics[key]["acc1"] = baseline_acc1
            baseline_metrics[key]["acc2"] = baseline_acc2

            predicted_metrics[key] = {}
            predicted_metrics[key]["acc1"] = predicted_acc1
            predicted_metrics[key]["acc2"] = predicted_acc2

    baseline_df = pd.DataFrame.from_dict(baseline_metrics, orient="index")
    predicted_df = pd.DataFrame.from_dict(predicted_metrics, orient="index")

    df = baseline_df.merge(
        predicted_df,
        left_index=True, right_index=True,
        suffixes=("_baseline", "_predicted")
    )

    df["acc1_baseline"] = df["acc1_baseline"].astype(float)
    df["acc1_predicted"] = df["acc1_predicted"].astype(float)
    df["acc2_baseline"] = df["acc2_baseline"].astype(float)
    df["acc2_predicted"] = df["acc2_predicted"].astype(float)

    print("saving .csv file")
    df.to_csv(DATA_FOLDER, "ballroom_metrics.csv", index=False)

    return


def main(model_path, kmin, kmax):
    model = tf.keras.models.load_model(model_path)
    ballroom = loader.custom_dataset_loader(
        path=DATASET_FOLDER, dataset_name="ballroom", folder="")

    # TODO:
    # 0. check if file with tempograms exist so we don't need to calculate it
    # 1. calibrate model
    a, b, quad = calibrate_model(model, kmin, kmax, ballroom)

    # evaluation_file = os.path.join(DATA_FOLDER, "evaluation", "ballroom_metrics.h5")
    evaluation_file = os.path.join(DATA_FOLDER, "ballroom_evaluation.h5")
    main_file = os.path.join(DATA_FOLDER, "ballroom.h5")

    evaluate_model(ballroom, evaluation_file, model, a, b, quad)

    get_metrics(evaluation_file)

    return


if __name__ == "__main__":
    import fire
    fire.Fire(main)
