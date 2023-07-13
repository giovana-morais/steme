"""
Create new dataset doing augmentations directly in the tempogram domain
"""
import h5py
import numpy as np

from steme import dataset, data_augmentation, paths


if __name__ == "__main__":
    # 1. load GTZAN
    gtzan, tracks, tempi = dataset.gtzan_data()

    main_file = "gtzan_tempogram_aug"

    main_filepath = os.path.join(paths.DATA_FOLDER, f"{main_file}.h5")

    linear_theta = np.arange(30, 670, 1)

    with h5py.File(main_filepath) as hf:
        for track in tracks:
            audio, sr = gtzan.track(track_id).audio

            # 2. calcular todos os tempogramas com o dobro do tamanho
            T, freqs, times = audio.tempogram(audio, sr, window_size_in_seconds=10,
                    t_type="fourier", theta=linear_theta)

            # 3. reduzir o tamanho de geral
            aug_T = get_even_rows(T)
            aug_bpm = gtzan.track(track_id).bpm / 2

            # 4. salvar os tempogramas reduzidos + o novo andamento (que será metade do
            # original)
            hf.create_dataset(f"{track_id}_augmented", data=T)

            # save tempo
            with open(os.path.join(DATASET_PATH,
                f"{track_id}_augmented.bpm"), "w") as f:
                f.write(str(aug_bpm))
