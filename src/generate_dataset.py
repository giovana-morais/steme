import os
import random

import h5py

import audio
import dataset
import loader
from paths import *

def main(dataset_name, dataset_type, synthetic, tmin, bins_per_octave, n_bins,
        t_type, **kwargs):
    theta = dataset.variables_non_linear(tmin=tmin,
            bins_per_octave=bins_per_octave,
            n_bins=n_bins)

    if synthetic == False:
        dataset.generate_dataset(dataset_name, dataset_type, theta)
    else:
        dataset.generate_synthetic_dataset(dataset_name=dataset_name,
                dataset_type=dataset_type, theta=theta,
                t_type=t_type, **kwargs)

    return

if __name__ == "__main__":
    import fire
    fire.Fire(main)
