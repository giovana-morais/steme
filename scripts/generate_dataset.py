import steme.audio as audio
import steme.dataset as dataset
from steme.paths import *


def main(dataset_name, dataset_type, synthetic, tmin, bins_per_octave, n_bins,
         t_type, **kwargs):
    theta = dataset.variables_non_linear(tmin=tmin,
                                         bins_per_octave=bins_per_octave,
                                         n_bins=n_bins)

    if not synthetic:
        dataset.generate_dataset(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            theta=theta,
            t_type=t_type)
    else:
        dataset.generate_synthetic_dataset(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            theta=theta,
            t_type=t_type,
            **kwargs)

    return


if __name__ == "__main__":
    import fire
    fire.Fire(main)
