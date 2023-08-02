# -*- coding: utf-8 -*-
import os

import h5py
import keras
import numpy as np
import tensorboard
import tensorflow as tf

import steme.dataset as dt
import steme.models as models
from steme.paths import *

LEARNING_RATE = 10e-4


def train_model(
        model,
        model_name,
        train_data,
        validation_data,
        epochs,
        early_stopping):

    log_dir = os.path.join(LOG_FOLDER, f"{model_name}")  # _{TIMESTAMP}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    model_path = os.path.join(MODEL_FOLDER, f"{model_name}")  # _{TIMESTAMP}")

    if not os.path.isdir(model_path):
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            initial_value_threshold=None,
        )

        callbacks = [tensorboard_callback, model_checkpoint]

        if early_stopping:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10)
            callbacks.append(early_stopping_callback)

        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
        model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks
        )
    else:
        print(f"Model exists. Loading {model_name}")
        model = tf.keras.models.load_model(model_path)

    return model


def main(model_name, epochs, early_stopping, main_file,
         kmin, kmax, tmin, n_bins, bins_per_octave, sigma_type, model_type,
         w_tempo=10e4, w_recon=1, **kwargs):

    main_filepath = os.path.join(DATA_FOLDER, f"{main_file}.h5")

    response = dt.read_dataset_info(main_file)

    train_filepath = response["train_filepath"]
    validation_filepath = response["validation_filepath"]
    train_setsize = response["train_setsize"]
    validation_setsize = response["validation_setsize"]

    training_tmin = response["tmin"]
    training_tmax = response["tmax"]

    sigma = dt.sigma(training_tmin, training_tmax, bins_per_octave)

    print(f"sigma = {sigma}")

    if model_type == "spice":
        model = models.spice(sigma, w_tempo, w_recon)
    else:
        model = models.convolutional_autoencoder(sigma, w_tempo, w_recon)

    output_signature = (
        # input shapes
        (
            tf.TensorSpec(shape=(128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.float32)
        ),
        # output shapes
        (
            tf.TensorSpec(shape=(128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.float32),
            tf.TensorSpec(shape=(1), dtype=tf.float32)
        )
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: dt.tempo_data_generator(train_filepath,
                                        set_size=train_setsize,
                                        kmin=kmin,
                                        kmax=kmax
                                        ),
        output_signature=output_signature
    )
    train_dataset = train_dataset.batch(64)

    validation_dataset = tf.data.Dataset.from_generator(
        lambda: dt.tempo_data_generator(validation_filepath,
                                        set_size=validation_setsize,  # <- synthetic data
                                        kmin=kmin,
                                        kmax=kmax
                                        ),
        output_signature=output_signature
    )
    validation_dataset = validation_dataset.batch(64)

    model = train_model(
        model=model,
        model_name=model_name,
        train_data=train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        early_stopping=early_stopping)
    return


if __name__ == "__main__":
    import fire
    fire.Fire(main)
