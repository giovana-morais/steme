"""
Augment dataset in audio domain
"""

import logger
import os
import random
from typing import List, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pyrubberband

import steme.augmentation as aug
import steme.dataset as dataset
import steme.path as paths

DATASET_PATH = os.path.join(paths.DATASET_FOLDER, "gtzan_augmented_log")

def remove_tracks(to_remove):
    for track_id in to_remove:
        try:
            logger.debug(f"removing {track_id}")
            os.remove(os.path.join(DATASET_PATH, f"audio/{track_id}.wav"))
            os.remove(os.path.join(DATASET_PATH, f"annotations/tempo/{track_id}.bpm"))
        except:
            logger.debug("already removed")
            continue
    return

if __name__ == "__main__":
    gtzan, tracks, tempi = dataset.gtzan_data()
    dist_low = dataset.lognormal70()
    theta = dataset.variables_non_linear(25, 40, 190)
    log_bins = theta[(theta > 30) & (theta < 370)][::2]
    bins = log_bins
    diff_tempi = dist_low_hist[0] - gtzan_dist[0]

    helper_dict = aug.create_helper_dict(finer_bins)
    gtzan_mapping = {}

    for i in tracks:
        tempo = gtzan.track(i).tempo

        boundaries = np.digitize(tempo, finer_bins)
        gtzan_mapping[i] = (tempo, f"{finer_bins[boundaries-1]}, {finer_bins[boundaries]}")
        helper_dict[f"{finer_bins[boundaries-1]}, {finer_bins[boundaries]}"].append(i)

    transformation_dict = aug.reset_transformation_dict()

	to_remove = []
	j = 0
	for key, val in list(transformation_dict.items())[::-1]:
	# for key, val in transformation_dict.items():
		if val < 0:
			logger.info(f"augmenting tracks from {key}")
			for track_id in helper_dict[key]:
				logger.debug(track_id)
				original_tempo = gtzan.track(track_id).tempo
				original_boundaries = gtzan_mapping[track_id][1]

				str_boundaries = aug.check_missing_tracks(transformation_dict)

				if str_boundaries is None or key == str_boundaries[0]:
		            logger.debug(transformation_dict)
					break

				new_tempo_boundaries = aug.key_boundaries(str_boundaries[0])

				# if key == str_boundaries[0]:
				# 	print(f"we will not transform {key} into {str_boundaries[0]}")
	# #                 transformation_dict[str_boundaries[0]] -= 1
				# 	break

				new_tempo = random.uniform(float(new_tempo_boundaries[0]), float(new_tempo_boundaries[1]))

				tempo_rate = new_tempo/original_tempo

				x, fs = gtzan.track(track_id).audio
				to_remove.append(track_id)

	            logger.debug(f"original_tempo {original_tempo}, new_tempo {new_tempo}, tempo_rate {tempo_rate}")

				# pyrubberband parameters
                # -3 means the finest algorithm, therefore the best audio
                # quality
				rbags = {"-3": ""}
				x_stretch = pyrb.time_stretch(x, fs, tempo_rate, **rbags)

                # update dicts
				transformation_dict[str_boundaries[0]] -= 1
				transformation_dict[original_boundaries] += 1

				# save audio
				sf.write(os.path.join(DATASET_PATH, f"audio/{track_id}_augmented.wav"), x_stretch, fs, subtype="PCM_24")

				# save tempo
				with open(os.path.join(DATASET_PATH, f"annotations/tempo/{track_id}_augmented.bpm"), "w") as f:
					f.write(str(new_tempo))
