"""
Augmentation helper functions
"""

import logging
from typing import List, Dict, Tuple

import numpy as np
import numpy.typing as npt


def key_boundaries(key: str) -> List[int]:
    """
    transform a string key of type '60, 65' into a list of ints [60, 65]
    """
    return [int(i) for i in key.split(", ")]


def check_missing_tracks(transformation_dict: Dict[str, float]) -> Tuple[str,
        float]:
    """
    return the first occurrence of a boundary that needs to be filled with
    data.
    """
    for k, v in transformation_dict.items():
        if v > 0:
            return k, v


def create_transformation_dict(verbose: bool = True) -> Dict[str, float]:
	"""
		create dictionary with boundaries and the augmentation needed in
        that boundary. e.g. {"[60, 65]": -2" means that 2 tracks need to be
        removed from the interval [60,65].
	"""
    removals = 0
    additions = 0

    transformation_dict = {}

    for idx, value in enumerate(diff_tempi):
        transformation_dict[f"{finer_bins[idx]}, {finer_bins[idx+1]}"] = value

        if value < 0:
            message = f"remove {value} samples"
            removals += np.abs(value)
        elif value > 0:
            message = f"add {value} samples"
            additions += value
        else:
            message = "do nothing"

        logger.debug(f"{finer_bins[idx]} - {finer_bins[idx+1]}: {message}")

    logger.info(f"total removals = {removals}, total additions = {additions}")

    return transformation_dict


def reset_transformation_dict() -> Dict[str, float]:
    """
    return a new transformation dict
    """
    return create_transformation_dict(verbose=False)


def tempogram_augmentation():
    """
    create new tempogram with half the tempo of the original

    parameters
    ---
    T : np.array (2D)
        tempogram matrix

    return
    ---
    augmented_T : np.array (2D)

    """
    return


# TODO: arrumar o nome dessa função pq tá tenebroso
def get_even_rows(T: npt.ArrayLike) -> npt.ArrayLike:
    """
    create tempogram with only the even lines of the input tempogram

    parameters
    ---
    T : np.array (2D)
        tempogram matrix

    return
    ---
    augmented_T : np.array (2D)

    """

    if T.shape[0] % 2 != 0:
        raise ValueError(f"Tempogram shape is not suitable for this reduction. \
        Expected even rows, but got {T.shape}")

    augmented_T = T[::2, :].copy()

    return augmented_T


def create_helper_dict(bins: list) -> Dict[str, list(str)]:
    # criar um dicionário com intervalo: {track_ids}
    # [30,40]: ["classical.0000", "blues.0010"]
    helper_dict = {}
    for idx in range(len(bins)-1):
        helper_dict[f"{bins[idx]}, {bins[idx+1]}"] = []

    return helper_dict
