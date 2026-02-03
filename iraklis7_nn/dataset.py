from pathlib import Path
from loguru import logger
import tensorflow_datasets as tfds
import typer

import iraklis7_nn.config as config
import iraklis7_nn.dataset as dataset
import iraklis7_nn.plots as plots

app = typer.Typer()


def make_dataset_raw():
    try:
        return tfds.load(
            "mnist",
            # split=['train', 'test'],
            split="all",
            # shuffle_files=True,
            as_supervised=True,
            # batch_size=-1,
            with_info=True,
        )
    except Exception as e:
        logger.exception("Unable to load dataset: " + str(e))
        raise


def make_dataset():
    try:
        return tfds.load(
            "mnist",
            # split=['train', 'test'],
            split=[
                "train[20%:]+test[20%:]",
                "train[0%:10%]+test[0%:10%]",
                "train[10%:20%]+test[10%:20%]",
            ],
            shuffle_files=False,
            as_supervised=True,
            with_info=True,
        )
    except Exception as e:
        logger.exception("Unable to load dataset: " + str(e))
        raise

def make_dataset_random():
    try:
        return tfds.load(
            "mnist",
            # split=['train', 'test'],
            split=[
                "train[20%:]+test[20%:]",
                "train[0%:10%]+test[0%:10%]",
                "train[10%:20%]+test[10%:20%]",
            ],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
    except Exception as e:
        logger.exception("Unable to load dataset: " + str(e))
        raise

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = config.RAW_DATA_DIR,
    output_path: Path = config.RAW_DATA_DIR,
    figures_path: Path = config.FIGURES_DIR,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    ds, ds_info = dataset.make_dataset_raw()
    logger.debug(ds_info)

    df = tfds.as_dataframe(ds=ds, ds_info=ds_info)
    logger.info(f"The dataset contains {len(df)} images and their corresponding labels")

    df_samples = df.sample(30)
    # Index needs to be reset for plots.plot_bmp_grid to work
    df_samples = df_samples.reset_index()

    # Visualize a few samples from the dataset
    sample_path = figures_path.joinpath(config.SAMPLE_PLOT)
    plots.plot_bmp_grid(
        df_samples.image,
        df_samples.label,
        predictions=None,
        columns=6,
        height=8,
        width=8,
        show=False,
        output_path=sample_path,
    )

    # Save data set
    raw_path = output_path.joinpath(config.RAW_SET)
    ds.save(str(raw_path))

    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
