from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import tensorflow as tf
import typer

import iraklis7_nn.config as config
import iraklis7_nn.plots as plots

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = config.PROCESSED_DATA_DIR,
    output_path: Path = config.PROCESSED_DATA_DIR,
    figures_path: Path = config.FIGURES_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing analysis on model predictions...")

    # Load predixtions
    pred_path = input_path.joinpath(config.PRED_SET)
    y_pred_full = np.loadtxt(str(pred_path))
    y_pred = np.array(list(map(np.argmax, y_pred_full)))

    # Load test dataset
    ds_test_path = input_path.joinpath(config.TEST_SET)
    ds_test = tf.data.Dataset.load(str(ds_test_path))
    # Create a DataFrame for easier analysis
    y_test = [(example.numpy(), label.numpy()) for example, label in ds_test]
    df_test = pd.DataFrame(y_test, columns=["example", "label"])

    # Add column for mapping to training dataset
    def gen_map(data):
        batch_size = len(df_test["label"].values[0])
        batch_number = int(data / batch_size)
        batch_pos = data - (batch_number * batch_size)
        return (batch_number, batch_pos)

    # Create a dataframe that will hold all mis-classifications
    df_mcs = pd.DataFrame(y_pred, columns=["predictions"])
    df_mcs["prediction_list"] = list(np.argsort(-y_pred_full))
    df_mcs["map"] = df_mcs["predictions"].index.to_series().map(gen_map)
    df_mcs.sample(5)

    # Using the ds_test map, add a column containing the labels
    def find_label(data):
        batch_number = data[0]
        batch_pos = data[1]
        return df_test["label"].values[batch_number][batch_pos]

    df_mcs["labels"] = df_mcs["map"].map(find_label)
    df_mcs.sample(5)

    # Compare predictions and labels and drop matching values
    df_mcs.drop(df_mcs[df_mcs.predictions == df_mcs.labels].index, inplace=True)
    print(f"Total number of mis-classifications: {len(df_mcs)}")
    print(df_mcs.head(5))

    # Using the ds_test map, add a column containing the images
    # We do this at this point, where the dataframe is at its smallest size
    def find_images(data):
        batch_number = data[0]
        batch_pos = data[1]
        return df_test["example"].values[batch_number][batch_pos]

    # Create a dataframe with samples from df_mcs
    # Index needs to be reset for plots.plot_bmp_grid to work
    df_sample = df_mcs.sample(30)
    df_sample["images"] = df_sample["map"].map(find_images)
    df_sample = df_sample.reset_index()
    df_sample.head()

    # Visualize plot
    plot_path = figures_path.joinpath(config.SAMPLE_MC_PLOT)
    plots.plot_bmp_grid(
        df_sample.images,
        df_sample.labels,
        df_sample.predictions,
        columns=6,
        height=8,
        width=8,
        show=False,
        output_path=plot_path,
    )

    # Use SQL like queries for clarity
    import duckdb

    # Count mis-classifications per label
    df_mcs_totals = duckdb.query(
        "SELECT labels, count(predictions) FROM df_mcs group by labels order by labels"
    ).df()
    logger.info(df_mcs_totals)
    import statistics

    logger.info(
        f"Mean of predictions count: {statistics.mean(df_mcs_totals['count(predictions)'])}"
    )

    # Visualize with bar plot
    plot_path = figures_path.joinpath(config.MCS_PER_LABEL_PLOT)
    plots.plot_xy_bar(
        df_mcs_totals.labels,
        df_mcs_totals["count(predictions)"],
        "Mis-classifications per label",
        show=False,
        output_path=plot_path,
    )

    # Distribution of mis-classifications per label
    df_mcs_pl = duckdb.query(
        "SELECT labels, predictions, count(predictions) FROM df_mcs group by labels, predictions order by labels, predictions"
    ).df()
    print(df_mcs_pl)

    # Visualize
    for i in range(10):
        query = f"SELECT labels, predictions, count(predictions), count(*) * 100.0 / sum(count(*)) over() as percentage FROM df_mcs where labels={i} group by labels, predictions order by labels, predictions"
        result = duckdb.query(query).df()
        logger.info(result)
        cpath = figures_path.joinpath(config.DIST_PLOT + f"{i}.png")
        plots.plot_pie(
            result.predictions,
            result["count(predictions)"],
            f"Distribution of mis-classified labels for {i}",
            show=False,
            output_path=cpath,
        )

    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
