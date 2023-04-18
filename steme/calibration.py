import random

import h5py
import numpy as np
import tensorflow as tf

import steme.audio as audio
import steme.dataset as dataset
import steme.utils as utils


def load_calibration_tracks(filename):
    with h5py.File(filename, "r") as hf:
        tracks = [key for key in hf.keys()]
        bpm_dict = {}

        for key, value in hf.items():
            reference_tempo = value["reference_tempo"][()]
            bpm_dict[reference_tempo] = {}

            bpm_dict[reference_tempo]["T"] = value["T"][:]
            bpm_dict[reference_tempo]["t"] = value["t"][:]
            bpm_dict[reference_tempo]["freqs"] = value["freqs"][:]
            bpm_dict[reference_tempo]["audio"] = value["audio"][:]
            bpm_dict[reference_tempo]["reference_tempo"] = value["reference_tempo"][:]

    return bpm_dict


def choose_calibration_candidates(calibration_range):
    if not isinstance(calibration_range, list):
        raise TypeError(
            f"calibration_range should be a list or an array, but it \
        is {type(calbration_range)}")
    gtzan, tracks, tempi = dataset.gtzan_data()

    gtzan_info = {}
    for i in tracks:
        gtzan_info[i] = gtzan.track(i).tempo

    gtzan_info = {
        k: v for k,
        v in sorted(
            gtzan_info.items(),
            key=lambda item: item[1])}

    calibration_tracks = {}
    for i in calibration_range:
        candidates = {
            k: v for k,
            v in gtzan_info.items() if np.abs(
                v - i) <= 1}

        random_candidate = random.choice(list(candidates.keys()))
        calibration_tracks.update(
            {random_candidate: candidates[random_candidate]})

    return calibration_tracks


def synthetic_tracks(theta, step):
    # tracks = theta[(theta > 30) & (theta < 300)][::step].copy()
    tracks = np.arange(30, 340, 10)
    bpm_dict = {}

    for i in tracks:
        bpm_dict[i] = {}
        bpm_dict["sr"] = 22050
        bpm_dict["audio"] = audio.click_track(bpm=i, sr=22050)

    return bpm_dict


def calibration_synthetic(
        model_name,
        model_path,
        tmin,
        n_bins,
        bins_per_octave,
        n_predictions,
        distribution,
        t_type):
    theta = dataset.variables_non_linear(
        tmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    bpm_dict = create_calibration_tracks(theta, step)

    model = tf.keras.models.load_model(model_path)

    for bpm, val in bpm_dict.items():
        T, t, bpms = audio.tempogram(
            val["audio"], val["sr"], window_size_seconds=10, t_type=t_type, theta=theta)

        bpm_dict[bpm] = {}
        bpm_dict[bpm]["T"] = T
        bpm_dict[bpm]["t"] = t
        bpm_dict[bpm]["freqs"] = bpms

    bpm_dict, a, b, _ = _calibrate(
        bpm_dict, model, kmin, kmax, n_predictions, fixed=True
    )
    bpm_preds = [
        v["predictions"] for k, v in bpm_dict.items()
    ]
    bpm_tracks = np.round(list(bpm_dict.keys()), 2)

    return bpm_preds


def _calibrate(bpm_dict, model, kmin, kmax, n_predictions=100, fixed=False):
    print("Calibrating model")
    model_output = np.zeros(len(bpm_dict.keys()))
    j = 0
    for bpm in bpm_dict.keys():
        T = bpm_dict[bpm]["T"]

        preds = np.zeros(n_predictions)
        step = T.shape[1] // n_predictions

        for i in range(n_predictions):
            slice_idx = i * step
            if fixed:
                s1, sh1, s2, sh2, _ = dataset.get_tempogram_slices(
                    T=T, kmin=kmin, kmax=kmax, shift_1=0, shift_2=0, slice_idx=slice_idx)
            else:
                s1, sh1, s2, sh2, _ = dataset.get_tempogram_slices(
                    T=T, kmin=kmin, kmax=kmax, slice_idx=slice_idx
                )
            s1 = s1[np.newaxis, :]
            s2 = s2[np.newaxis, :]

            # xhat1, xhat2, y1, y2 = model.predict([s1, s2, sh1, sh2], verbose=0)
            xhat1, xhat2, y1, y2 = model.predict([s1, s1, sh1, sh1], verbose=0)
            preds[i] = y1[0][0]
            # preds[i] = (y1[0][0]+y2[0][0])/2
            # preds[i] = y2[0][0]

        bpm_dict[bpm]["slice"] = s1[0, :, 0]
        bpm_dict[bpm]["shift"] = sh1
        bpm_dict[bpm]["estimation"] = xhat1[0, :, 0]
        bpm_dict[bpm]["predictions"] = np.array(preds)
        # print(f"Predictions for {bpm} = {np.array(preds)}")

        model_output[j] = np.median(np.array(preds))
        j += 1

    # quad = np.poly1d(np.polyfit(model_output, list(bpm_dict.keys()), 2))
    a, b = utils.get_slope(model_output, list(bpm_dict.keys()))

    return bpm_dict, a, b  # , quad
