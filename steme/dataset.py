import os
import random
from collections import Counter

import h5py
import numpy as np
import mirdata
from scipy.stats import tukeylambda, lognorm, uniform

import steme.audio as audio
import steme.loader as loader
import steme.paths as paths


def generate_biased_data(main_file, distribution, theta, t_type):
    """
    Generates synthetic data (click tracks) following a distribution.

    main_file : str
        filename that will be used to store the data
    distribution : dict
        array with BPM/frequency. each track is duplicated following the
        frequency of the BPM.
    theta : float
        parameter to calculate the tempogram
    tempogram_type : str
        tempogram type. options are "fourier", "autocorrelation", "hybrid"
    """
    with h5py.File(f"data/{main_file}.h5", "w") as hf:
        for track_id in distribution:
            sr = 22050
            x = audio.click_track(track_id, sr)

            if distribution[track_id] > 1:
                x = np.repeat(x, distribution[track_id])

            T, t, bpm = audio.tempogram(x, sr, window_size_seconds=10,
                                        t_type=t_type, theta=theta)

            hf.create_dataset(str(track_id), data=T)

    return


def remove_out_of_bound_data(distribution):
    """
    Removes data below 30 BPM and above 350 BPM
    """
    key_to_remove = [k for k, v in distribution.items() if k < 30 or k > 350]
    new_dist = distribution.copy()

    for k in key_to_remove:
        new_dist.pop(k)

    return new_dist


def copy_data(main_file, new_file, ids):
    """
    Copy a specific list of ids to a new file
    """
    samples = 0
    with h5py.File(f"data/{main_file}.h5", "r") as rhf:
        with h5py.File(f"data/{new_file}.h5", "w") as whf:
            for track_id in ids:
                tempogram = rhf.get(str(track_id))
                samples += tempogram.shape[1]

                whf.create_dataset(str(track_id), data=tempogram)
    return samples


def gtzan_data():
    gtzan = mirdata.initialize(
        "gtzan_genre",
        data_home=os.path.join(
            paths.DATASET_FOLDER,
            "gtzan_genre"),
        version="default")
    tracks = gtzan.track_ids

    # remove tracks with no tempo annotation
    tracks.remove("reggae.00086")
    tempi = [gtzan.track(track_id).tempo for track_id in tracks]


    return gtzan, tracks, tempi


def giant_steps_data():
    gs = loader.custom_dataset_loader(
        path=paths.DATASET_FOLDER,
        dataset_name="giantsteps-tempo-dataset",
        folder=""
    )
    tracks = gs.track_ids
    # remove tracks with no tempo annotation
    tracks.remove("3041381.LOFI")
    tracks.remove("3041383.LOFI")
    tracks.remove("1327052.LOFI")

    tempi = [gs.track(track_id).tempo for track_id in tracks]

    return gs, tracks, tempi


def ballroom_data():
    ballroom = loader.custom_dataset_loader(
        path=paths.DATASET_FOLDER,
        dataset_name="ballroom",
        folder=""
    )

    tempi = [ballroom.track(i).tempo for i in ballroom.track_ids]
    tracks = [i for i in ballroom.track_ids]
    return ballroom, tracks, tempi

def gtzan_augmented_data():
    gtzan_augmented = loader.custom_dataset_loader(
        path=paths.DATASET_FOLDER,
        dataset_name="gtzan_augmented",
        folder="",
    )

    tracks = gtzan_augmented.track_ids
    tracks.remove("reggae.00086")
    tempi = [gtzan_augmented.track(track_id).tempo for track_id
            in tracks]

    return gtzan_augmented, tracks, tempi


def gtzan_augmented_log_data():
    gtzan_augmented = loader.custom_dataset_loader(
        path=paths.DATASET_FOLDER,
        dataset_name="gtzan_augmented_log",
        folder="",
    )

    tracks = gtzan_augmented.track_ids
    tracks.remove("reggae.00086")
    tempi = [gtzan_augmented.track(track_id).tempo for track_id
            in tracks]

    return gtzan_augmented, tracks, tempi

def brid_data():
    brid = loader.custom_dataset_loader(
        path=paths.DATASET_FOLDER,
        dataset_name="brid",
        folder=""
    )

    tempi = [brid.track(i).tempo for i in brid.track_ids]
    tracks = [i for i in brid.track_ids]
    return brid, tracks, tempi


def get_metadata_file(main_file):
    return os.path.join(paths.DATA_FOLDER, f"{main_file}_metadata.h5")


def read_dataset_info(main_file):
    dataset_metadata = get_metadata_file(main_file)
    print(f"Reading metadata file {dataset_metadata}")
    response = {}

    with h5py.File(dataset_metadata, "r") as hf:
        response["main_file"] = hf.get("main_file")[()].decode("UTF-8")
        response["validation_file"] = hf.get("validation_file")[
            ()].decode("UTF-8")
        response["train_file"] = hf.get("train_file")[()].decode("UTF-8")
        response["main_filepath"] = hf.get("main_filepath")[()].decode("UTF-8")
        response["validation_filepath"] = hf.get("validation_filepath")[
            ()].decode("UTF-8")
        response["train_filepath"] = hf.get("train_filepath")[
            ()].decode("UTF-8")
        response["distribution"] = hf.get("distribution")[:]
        response["validation_setsize"] = hf.get("validation_setsize")[()]
        response["train_setsize"] = hf.get("train_setsize")[()]
        response["tmin"] = hf.get("tmin")[()]
        response["tmax"] = hf.get("tmax")[()]

    return response


def write_dataset_info(main_file, response):
    dataset_metadata = get_metadata_file(main_file)
    print(f"Creating metadata file: {dataset_metadata}")

    with h5py.File(dataset_metadata, "w") as hf:
        for k, v in response.items():
            hf.create_dataset(k, data=v)

    return response


def generate_synthetic_dataset(
        dataset_name,
        dataset_type,
        theta,
        t_type,
        lam,
        loc,
        scale,
        size):
    """
    Creates synthetic datasets that will follow a distribution
    """

    if dataset_type == "tukey_lambda":
        dist = tukeylambda.rvs(lam, loc=loc, scale=scale, size=size,
                               random_state=42)
    elif dataset_type == "lognorm":
        dist = lognorm.rvs(
            lam,
            loc=loc,
            scale=scale,
            size=size,
            random_state=42)
    elif dataset_type == "uniform":
        dist = uniform.rvs(loc=loc, scale=scale, size=1000, random_state=42)
    elif dataset_type == "gtzan_synthetic":
        _, _, dist = gtzan_data()
    elif dataset_type == "log_uniform":
        # tmin = theta[0]
        # tmax = theta[-1]
        tmin = 30
        tmax = 240

        dist = tmin * np.e**(np.random.rand(size) * np.log(tmax / tmin))

    main_file = dataset_name
    print(f"dataset_name = {dataset_name}")

    dist_counter = Counter(dist)
    dist_counter = remove_out_of_bound_data(dist_counter)

    train_file = f"{main_file}_train"
    validation_file = f"{main_file}_validation"

    main_filepath = os.path.join(paths.DATA_FOLDER, f"{main_file}.h5")
    train_filepath = os.path.join(paths.DATA_FOLDER, f"{main_file}_train.h5")
    validation_filepath = os.path.join(
        paths.DATA_FOLDER, f"{main_file}_validation.h5")

    keys = [k for k in dist_counter.keys()]
    tmin = min(keys)
    tmax = max(keys)

    print("Generating biased files")
    if not os.path.isfile(main_filepath):
        generate_biased_data(main_file, dist_counter, theta, t_type)
        random.shuffle(keys)

        train_split = int(len(keys) * 0.8)
        train_ids = keys[:train_split]
        validation_ids = keys[train_split:]

        # create train and validation files
        train_samples = copy_data(main_file, train_file, train_ids)
        validation_samples = copy_data(
            main_file, validation_file, validation_ids)

        print(f"total train samples: {train_samples}")
        print(f"total validation samples: {validation_samples}")

        response = {
            "distribution": dist,
            "main_file": main_file,
            "train_file": train_file,
            "validation_file": validation_file,
            "main_filepath": main_filepath,
            "train_filepath": train_filepath,
            "validation_filepath": validation_filepath,
            "train_setsize": train_samples,
            "validation_setsize": validation_samples,
            "tmin": tmin,
            "tmax": tmax,
        }

        write_dataset_info(main_file, response)
    else:
        response = read_dataset_info(main_file)

    return response

def lognormal70():
    return lognorm.rvs(0.25, loc=30, scale=50, size=1000, random_state=42)

def lognormal150():
    return lognorm.rvs(0.25, loc=70, scale=50, size=1000, random_state=42)

def lognormal170():
    return lognorm.rvs(0.25, loc=120, scale=50, size=1000, random_state=42)

def log_uniform():
    return 30*np.e**(np.random.rand(1000)*np.log(240/30))

def uniform():
    return uniform.rvs(30, scale=210,size=1000, random_state=42)

def generate_dataset(dataset_name, dataset_type, theta, t_type):
    if dataset_type == "gtzan":
        gtzan, tracks, tempi = gtzan_data()
    elif dataset_type == "gtzan_augmented":
        gtzan, tracks, tempi = gtzan_augmented_data()
    elif dataset_type == "gtzan_augmented_log":
        gtzan, tracks, tempi = gtzan_augmented_log_data()
    elif dataset_type == "giant_steps":
        gs, tracks, tempi = giant_steps_data()
    elif dataset_type == "ballroom":
        ballroom, tracks, tempi = ballroom_data()
    elif dataset_type == "gtzan_giant_steps":
        gtzan, gtzan_tracks, gtzan_tempi = gtzan_data()
        gs, gs_tracks, gs_tempi = giant_steps_data()
        tracks = gtzan_tracks + gs_tracks
        tempi = gtzan_tempi + gs_tempi
    elif dataset_type == "brid":
        brid, tracks, tempi = brid_data()
    elif dataset_type == "gtzan+giant_steps":
        gtzan, gtzan_tracks, gtzan_tempi = gtzan_data()
        gs, gs_tracks, gs_tempi = giant_steps_data()

        tracks = gtzan_tracks + gs_tracks
        tempi = gtzan_tempi + gs_tempi

    main_file = f"{dataset_name}"
    train_file = f"{main_file}_train"
    validation_file = f"{main_file}_validation"

    main_filepath = os.path.join(paths.DATA_FOLDER, f"{main_file}.h5")
    train_filepath = os.path.join(paths.DATA_FOLDER, f"{main_file}_train.h5")
    validation_filepath = os.path.join(
        paths.DATA_FOLDER, f"{main_file}_validation.h5")

    tmin = min(tempi)
    tmax = max(tempi)

    print(f"Generating tempogram files: {main_file}.h5")
    if not os.path.isfile(main_filepath):
        with h5py.File(main_filepath, "w") as hf:
            for track_id in tracks:
                if "LOFI" in track_id:
                    x, sr = gs.track(track_id).audio
                else:
                    x, sr = gtzan.track(track_id).audio

                T, t, bpm = audio.tempogram(
                    x, sr, window_size_seconds=10, t_type=t_type, theta=theta)

                hf.create_dataset(str(track_id), data=T)

        random.shuffle(tracks)

        train_split = int(len(tracks) * 0.8)
        train_ids = tracks[:train_split]
        validation_ids = tracks[train_split:]

        # create train and validation files
        train_samples = copy_data(main_file, train_file, train_ids)
        validation_samples = copy_data(
            main_file, validation_file, validation_ids)

        response = {
            "distribution": tempi,
            "main_file": main_file,
            "train_file": train_file,
            "validation_file": validation_file,
            "main_filepath": main_filepath,
            "train_filepath": train_filepath,
            "validation_filepath": validation_filepath,
            "train_setsize": train_samples,
            "validation_setsize": validation_samples,
            "tmin": tmin,
            "tmax": tmax,
        }
        write_dataset_info(main_file, response)
    else:
        response = read_dataset_info(main_file)

    return response


def sigma(tmin, tmax, bins_per_octave):
    """
    Calculates sigma for a given shift interval.
    """
    if tmin > tmax:
        raise ValueError(f"tmin > tmax. {tmin, tmax}")

    sigma = 1 / (bins_per_octave * np.log2(tmax / tmin))

    return sigma


def sigma_diff(kmin, kmax):
    return 1 / (kmax - kmin)


def get_tempogram_slices(
        T,
        F=128,
        slice_idx=None,
        kmin=0,
        kmax=8,
        shift_1=None,
        shift_2=None):
    """
    Return a F-dimension slice from the tempogram

    Parameters
    ---------
    T : np.array
        Tempogram
    F : int, optional
        Size of the slice returned. Default is 128.
        If F > T.shape[0], raises an exception.
    slice_idx : int, optional
        Slice position you want to return. Default is None and returns a random
        slice. If slice_idx != None, returns specifically the slice_idx
        position.
    Return
    ------
    tempo_sample_1, tempo_sample_2 : np.ndarray(1,F)
    shift_1, shift_2 : int
        Integers representing the shifts

    """
    if T.shape[0] < F:
        raise ValueError(
            f"Dimensions mismatch. It is not possible to retrieve a {F}-slice from a {T.shape} matrix")

    if shift_1 is None and shift_2 is None:
        shift_1 = random.randrange(kmin, kmax)
        shift_2 = random.randrange(kmin, kmax)

    if slice_idx is None:
        slice_idx = random.randint(0, T.shape[1] - 1)

    tempo_sample_1 = T[shift_1:shift_1 + F, slice_idx].copy()
    tempo_sample_2 = T[shift_2:shift_2 + F, slice_idx].copy()

    tempo_sample_1 = tempo_sample_1 / (tempo_sample_1.max() + 1e-6)
    tempo_sample_2 = tempo_sample_2 / (tempo_sample_2.max() + 1e-6)

    # correct array dimension for training
    tempo_sample_1 = tempo_sample_1[:, np.newaxis]
    tempo_sample_2 = tempo_sample_2[:, np.newaxis]

    shift_1 = np.array([shift_1])
    shift_2 = np.array([shift_2])

    return tempo_sample_1, shift_1, tempo_sample_2, shift_2, slice_idx


def tempo_data_generator(filename, set_size=12000, **kwargs):
    """
    Parameters
    ----------
    filename : str
        The file path
    set_size : int, optional
        Total number of samples in training/test set
    **kwargs , optional
        Parameters of get_tempogram_slices function

    Returns
    -------
    (sample_1, sample_2, shift_1, shift_2), (sample_1, sample_2, shift_1, shift_2)
        Two tuples with both inputs and outputs.
    sample_1, sample_2 : np.array
        np.array containing tempogram from the test file
    shift_1, shift_2 : int
        The shift made in the representation to calculate the tempo
    """

    if not os.path.isfile(filename):
        raise ValueError(f"File '{filename}' does not exist")

    if not h5py.is_hdf5(filename):
        raise ValueError(f"File '{filename}' is not an HDF5 file")

    if set_size <= 0:
        raise ValueError(f"Invalid set size {set_size}")

    with h5py.File(filename, "r") as hf:
        track_ids = [key for key in hf.keys()]
        for i in range(set_size):
            track_id = random.choice(track_ids)
            tempogram = hf.get(track_id)

            tempogram_1, shift_1, tempogram_2, shift_2, slice_idx = get_tempogram_slices(
                tempogram, **kwargs)

            yield (tempogram_1, tempogram_2, shift_1, shift_2), (tempogram_1, tempogram_2, shift_1, shift_2)


def variables_2bpm():
    theta = np.arange(30, 350, 2)
    kmin = 0
    kmax = 16
    return theta, kmin, kmax


def variables_non_linear(tmin=25, bins_per_octave=40, n_bins=190):
    frequencies = 2.0 ** (np.arange(0, n_bins, dtype=float) / bins_per_octave)
    theta = tmin * frequencies

    return theta
