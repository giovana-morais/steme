import os
import pickle
import random

import h5py
import numpy as np
import tensorflow as tf

import steme.audio as audio
import steme.dataset as dataset
import steme.calibration as calibration

def default_variables():
    return {
            "tmin": 25,
            "n_bins": 190,
            "bins_per_octave": 40,
            "kmin": 11,
            "kmax": 19
    }
def read_dataset_info(main_file):
    dataset_metadata = os.path.join("/home/gigibs/Documents/steme/data", f"{main_file}_metadata.h5")
    print(f"Reading metadata file {dataset_metadata}")
    response = {}

    with h5py.File(dataset_metadata, "r") as hf:
        response["main_file"] = hf.get("main_file")[()].decode("UTF-8")
        response["validation_file"] = hf.get("validation_file")[()].decode("UTF-8")
        response["train_file"] = hf.get("train_file")[()].decode("UTF-8")
        response["main_filepath"] = hf.get("main_filepath")[()].decode("UTF-8")
        response["validation_filepath"] = hf.get("validation_filepath")[()].decode("UTF-8")
        response["train_filepath"] = hf.get("train_filepath")[()].decode("UTF-8")
        response["distribution"] = hf.get("distribution")[:]
        response["validation_setsize"] = hf.get("validation_setsize")[()]
        response["train_setsize"] = hf.get("train_setsize")[()]
        response["tmin"] = hf.get("tmin")[()]
        response["tmax"] = hf.get("tmax")[()]

    return response

# def get_center_bins(step, offset):
# 	# defining the center bins for the random tracks in calibration
# 	# step = 5
# 	# offset = 5
# 	left = theta[(theta > 30) & (theta < 350)][::step]
# 	center = theta[(theta > 30) & (theta < 350)][offset::step]
# 	right = theta[(theta > 30) & (theta < 350)][offset::step]

# 	bins = []
# 	for i, j, k in zip(left, center, right):
# 		print(f"boundaries for {np.round(j,2)}: [{np.round(np.sqrt(i*j),2)}, {np.round(np.sqrt(j*k))}]")
# 		bins.append(i)
# 		bins.append(j)
#     return bins

def create_center_dict(n_predictions, tracks_per_bin):
	# n_predictions = 2
	# tracks_per_bin = 50

	center_dict = {}
	for idx, val in enumerate(center):
		left_boundary = np.sqrt(left[idx]*center[idx])
		right_boundary = np.sqrt(center[idx]*right[idx])

		center_dict[val] = np.random.uniform(left_boundary, right_boundary, size=tracks_per_bin)
	return center_dict

def calibration_results(dists, t_types, variation, n_predictions):
    results_dict = {}
    for dist_name in dists:
        results_dict[dist_name] = {}
        for t_type in t_types:
            print(dist_name, t_type)
            dataset_name = f"{dist_name}_{t_type}"

            response = read_dataset_info(dataset_name)
            distribution = response["distribution"]

            results_dict[dist_name][t_type] = {}

            model_name = f"{dataset_name}_15_default"
            model_path = f"../models/{variation}/{model_name}"

            model = tf.keras.models.load_model(model_path)

            for idx, val in enumerate(center):
                results_dict[dist_name][t_type][val] = {}
                sr = 22050
                preds = np.zeros(n_predictions*tracks_per_bin)

                j = 0

                for bpm in center_dict[val]:
                    x = audio.click_track(bpm=bpm, sr=sr)
                    T, t, bpms = audio.tempogram(x, sr, window_size_seconds=10, t_type=t_type, theta=theta)

                    step = T.shape[1]//n_predictions

                    for i in range(n_predictions):
                        slice_idx = i*step
                        s1, sh1, s2, sh2, _ = dataset.get_tempogram_slices(
                            T=T, kmin=kmin, kmax=kmax, shift_1=0, shift_2=0, slice_idx=slice_idx
                        )

                        s1 = s1[np.newaxis, :]

                        xhat1, xhat2, y1, y2 = model.predict([s1, s1, sh1, sh1], verbose=0)
                        preds[j] = y1[0][0]
                        j += 1
                results_dict[dist_name][t_type][val]["predictions"] = np.array(preds)
#                 results_dict[dist_name][t_type][val]["tracks"] = bpm_tracks

            del model
        return results_dict


if __name__ == "__main__":
    variables = default_variables()
    tmin = variables["tmin"]
    n_bins = variables["n_bins"]
    bins_per_octave = variables["bins_per_octave"]
    kmin, kmax = variables["kmin"], variables["kmax"]
    theta = dataset.variables_non_linear(tmin, n_bins=n_bins, bins_per_octave=bins_per_octave)

    step = 5
    offset = 5
    left = theta[(theta > 30) & (theta < 350)][::step]
    center = theta[(theta > 30) & (theta < 350)][offset::step]
    right = theta[(theta > 30) & (theta < 350)][offset::step]
    # bins = get_center_bins(5, 5)

    dists = [
            "gtzan_augmented_log_25_190_40",
            "gtzan_augmented_log_cropped_25_190_40",
            "log_uniform_25_190_40",
            "synthetic_lognorm_0.7_30_50_1000_25_190_40",
            "synthetic_lognorm_0.7_70_50_1000_25_190_40",
            "synthetic_lognorm_0.7_120_50_1000_25_190_40",
            "gtzan_25_190_40"
    ]

    t_types = ["fourier", "autocorrelation", "hybrid"]

    variations = ["early_stopping", "wo_early_stopping"]
    n_predictions = 1
    tracks_per_bin = 1

    center_dict = create_center_dict(n_predictions, tracks_per_bin)
    with open('center_dict_aug_full.pkl', 'wb') as f:
        pickle.dump(center_dict, f)

    for v in variations:
        results_dict = calibration_results(dists, t_types, v, n_predictions)
        with open(f'results_dict_aug_{variation}.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
