from pathlib import Path
from loguru import logger
import tensorflow as tf
import typer

import iraklis7_nn.config as config
import iraklis7_nn.dataset as dataset

app = typer.Typer()


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label

def optimise_ds(ds, ds_info, is_train=False):
    ds = ds.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    if(is_train):
        ds = ds.shuffle(ds_info.splits['train'].num_examples)
    ds = ds.batch(128)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = config.RAW_DATA_DIR,
    output_path: Path = config.PROCESSED_DATA_DIR,
    figures_path: Path = config.FIGURES_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")

    # Load the datasets for training, validation and test
    (ds_train, ds_val, ds_test), ds_info = dataset.make_dataset()
    logger.info(
        f"Train size: {len(ds_train)} Validation size: {len(ds_val)} Test size; {len(ds_test)}"
    )

    # Optimize datasets
    optimise_ds(ds_train, ds_info, is_train=True)
    optimise_ds(ds_val, ds_info, is_train=False)
    optimise_ds(ds_test, ds_info, is_train=False)
    
    # Save datasets to file
    ds_train_path = output_path.joinpath(config.TRAIN_SET)
    ds_val_path = output_path.joinpath(config.VAL_SET)
    ds_test_path = output_path.joinpath(config.TEST_SET)

    ds_train.save(str(ds_train_path))
    ds_val.save(str(ds_val_path))
    ds_test.save(str(ds_test_path))

    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
