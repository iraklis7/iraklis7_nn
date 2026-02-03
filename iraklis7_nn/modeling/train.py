from pathlib import Path
from loguru import logger
import tensorflow as tf
import typer

import iraklis7_nn.config as config

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = config.PROCESSED_DATA_DIR,
    output_path: Path = config.PROCESSED_DATA_DIR,
    figures_path: Path = config.FIGURES_DIR,
    models_path: Path = config.MODELS_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")

    # Load the datasets
    ds_train_path = input_path.joinpath(config.TRAIN_SET)
    ds_train = tf.data.Dataset.load(str(ds_train_path))
    ds_val_path = input_path.joinpath(config.VAL_SET)
    ds_val = tf.data.Dataset.load(str(ds_val_path))

    # Create the model
    model = tf.keras.models.Sequential(
        [
            # Need to get rid of warning by using Flatten and input_shape
            tf.keras.Input(shape=(28, 28)),
            #tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(
                128,
                activation="relu",
            ),
            tf.keras.layers.Dense(10),
        ]
    )

    # Print model summary
    model.summary()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Fit the model
    model.fit(
        ds_train,
        epochs=12,
        validation_data=ds_val,
    )

    # Save the model
    model_path = models_path.joinpath(config.MODEL_NAME)
    model.save(model_path)

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
