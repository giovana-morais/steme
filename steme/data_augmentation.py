from typing import List, Dict, Tuple


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
