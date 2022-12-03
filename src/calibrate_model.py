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


def calibrate_model_synthetic(model, kmin, kmax, theta, n_predictions):
    print("Creating tracks for calibration")
    bpm_tracks = create_calibration_tracks(theta, 10)
    bpm_dict = {}

    for bpm in bpm_tracks:
        sr = 22050
        x = audio.click_track(bpm=bpm, sr=sr)
        T, t, bpms = audio.tempogram(
            x, sr, window_size_seconds=10, t_type="hybrid", theta=theta)

        bpm_dict[bpm] = {}
        bpm_dict[bpm]["audio"] = x
        bpm_dict[bpm]["T"] = T
        bpm_dict[bpm]["t"] = t
        bpm_dict[bpm]["freqs"] = bpms

    bpm_dict, a, b, quad = _calibrate(
        bpm_dict=bpm_dict, model=model, kmin=kmin, kmax=kmax, n_predictions=n_predictions)

    return bpm_dict, bpm_tracks, a, b, quad


def calibrate_model_non_synthetic(model, kmin, kmax, theta, mirdata_dataset,
                                  n_predictions=100, interval=None):
    tempi = [mirdata_dataset.track(i).tempo for i in mirdata_dataset.track_ids]
    track = [i for i in mirdata_dataset.track_ids]
    tracks = []
    bpm_tracks = []

    if mirdata_dataset.name == "gtzan_genre":
        try:
            tempi.remove(None)
            track.remove("reggae.00086")
        except Exception as e:
            print(e)

    tempi = np.array(tempi)
    track = np.array(track)

    ordered_indexes = np.argsort(tempi)
    tempi = tempi[ordered_indexes]
    track = track[ordered_indexes]

    track_dict = {i: j for i, j in zip(tempi, track)}

    if interval is None:
        intervals = np.arange(30, 350, 10)
    else:
        intervals = interval

    bpm_dict = {}

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

            print(
                f"interval [{intervals[idx]}, {intervals[idx+1]}]: {track_dict[bpm]}")
            tracks.append(track_dict[bpm])
            bpm_tracks.append(bpm)
        except IndexError:
            print(
                f"no index for interval [{intervals[idx]}, {intervals[idx+1]}]")

    bpm_dict, a, b, quad = _calibrate(
        bpm_dict=bpm_dict, model=model, kmin=kmin, kmax=kmax, n_predictions=n_predictions)

    return bpm_dict, bpm_tracks, a, b, quad


def evaluate_model(ballroom, evaluation_file, model):
    print("Evaluating model")
    if not os.path.isfile(evaluation_file):
        print(f"Generating {evaluation_file}")
        with h5py.File(evaluation_file, "a") as whf:
            for track_id in ballroom.track_ids:
                print(track_id)
                predictions = []

                theta = dt.variables_non_linear()
                x, sr = ballroom.track(track_id).audio
                T, t, freqs = audio.tempogram(
                    x, sr, window_size_seconds=10, t_type="hybrid", theta=theta)
                reference_tempo = ballroom.track(track_id).tempo

                for i in range(T.shape[1]):
                    s1, sh1, s2, sh2, _ = dt.get_tempogram_slices(
                        T, slice_idx=i, kmin=11, kmax=19)

                    s1 = s1[np.newaxis, :]

                    xhat1, xhat2, y1, y2 = model.predict(
                        [s1, s1, sh1, sh1], verbose=0)
                    predictions.append(y1[0][0])

                predicted_tempo_linear = np.array(predictions) * a + b
                predicted_tempo_quadratic = quad(np.array(predictions))
                baseline_tempo = np.take(freqs, np.argmax(T, axis=-2))

                g = whf.create_group(track_id)
                g["reference_tempo"] = reference_tempo
                g["model_output"] = predictions
                # g["predicted_tempo_linear"] = predicted_tempo_linear
                # g["predicted_tempo_quadratic"] = predicted_tempo_quadratic
                g["baseline_tempo"] = baseline_tempo
                g["T"] = T.copy()
                g["t"] = t
                g["freqs"] = freqs
    else:
        print(f"{evaluation_file} already exists")

    return


def get_metrics(evaluation_file):
    print(f"Generating metrics for {evaluation_file}")
    baseline_metrics = {}
    predicted_metrics = {}

    print(f"{evaluation_file}")

    with h5py.File(f"{evaluation_file}", "r") as hf:
        for key, value in hf.items():
            baseline_tempo = value["baseline_tempo"][:]
            reference_tempo = value["reference_tempo"][()]
            predicted_tempo_linear = value["predicted_tempo_linear"][:]
            T = value["T"][:]
            t = value["t"][:]
            freqs = value["freqs"][:]

            baseline_acc1 = metrics.acc1(
                reference_tempo, np.median(baseline_tempo))
            baseline_acc2 = metrics.acc2(
                reference_tempo, np.median(baseline_tempo))

            predicted_acc1 = metrics.acc1(
                reference_tempo, np.median(predicted_tempo_linear))
            predicted_acc2 = metrics.acc2(
                reference_tempo, np.median(predicted_tempo_linear))

            baseline_metrics[key] = {}
            baseline_metrics[key]["reference_tempo"] = reference_tempo
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
    print(df.sample(5))
    print(f"{evaluation_file}.csv =====================")
    df.to_csv(f"{evaluation_file}.csv")

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

    # calibration on synthetic data
    synth_main_file = f"{model_name}_synth_data"
    bpm_dict_synth, bpm_tracks_synth, a_synth, b_synth, quad_synth = calibrate_model_synthetic(
        model, kmin, kmax, theta, n_predictions)
    utils.plot_reconstructions(
        bpm_tracks_synth,
        bpm_dict_synth,
        synth_main_file,
        theta)
    bpm_preds_synth = [v["predictions"] for k, v in bpm_dict_synth.items()]
    utils.plot_calibration(
        bpm_tracks_synth,
        bpm_preds_synth,
        distribution,
        synth_main_file)

    # calibration on ballroom data
    ballroom_main_file = f"{model_name}_ballroom_data"
    ballroom = loader.custom_dataset_loader(
        path=DATASET_FOLDER, dataset_name="ballroom", folder="")
    bpm_dict, bpm_tracks, a, b, quad = calibrate_model_non_synthetic(
        model, kmin, kmax, theta, ballroom, n_predictions=n_predictions)
    utils.plot_reconstructions(bpm_tracks, bpm_dict, ballroom_main_file, theta)
    bpm_preds = [v["predictions"] for k, v in bpm_dict.items()]
    utils.plot_calibration(
        bpm_tracks,
        bpm_preds,
        distribution,
        ballroom_main_file)

    # calibration on gtzan data
    # gtzan_main_file = f"{model_name}_gtzan_data"
    # gtzan, tracks, tempi = dt.gtzan_data()
    # bpm_dict_gtzan, bpm_tracks_gtzan, a_gtzan, b_gtzan, quad_gtzan = calibrate_model_non_synthetic(model, kmin, kmax,
    #         theta, gtzan, n_predictions=n_predictions,
    #         interval=np.arange(90,190,5))
    # utils.plot_reconstructions(bpm_tracks_gtzan, bpm_dict_gtzan, gtzan_main_file, theta)
    # bpm_preds_gtzan = [v["predictions"] for k, v in bpm_dict_gtzan.items()]
    # utils.plot_calibration(bpm_tracks_gtzan, bpm_preds_gtzan, tempi, gtzan_main_file)

    # evaluate on ballroom
    # get_metrics(evaluation_file)

    return


if __name__ == "__main__":
    import fire
    fire.Fire(main)
