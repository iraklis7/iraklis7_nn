from pathlib import Path
from loguru import logger
import numpy as np
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
    logger.info("Performing inference for model...")

    # Load model from file
    model_path = models_path.joinpath(config.MODEL_NAME)
    model = tf.keras.models.load_model(model_path)

    # Load test dataset
    ds_test_path = input_path.joinpath(config.TEST_SET)
    ds_test = tf.data.Dataset.load(str(ds_test_path))

    # Make predictions on test dataset
    y_pred_full = model.predict(ds_test)

    # Report on metrics
    model.get_metrics_result()

    # Save predictions to file
    pred_path = output_path.joinpath(config.PRED_SET)
    np.savetxt(str(pred_path), y_pred_full)

    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
