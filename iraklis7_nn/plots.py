from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import typer

from iraklis7_nn.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def plot_pie(x, y, title, show=False, output_path=None):
    ax = plt.axes()
    ax.pie(y, labels=x, autopct="%1.2f%%")
    ax.set_title(title)

    logger.info(f"Saving plot to: {output_path}")
    if output_path is not None:
        try:
            plt.savefig(output_path)
        except Exception as e:
            logger.exception("Unable to save plot: " + str(e))
    if show:
        plt.show()


def plot_xy_bar(x, y, title, show=False, output_path=None):
    ax = plt.axes()
    ax.bar(x, y, edgecolor="black")
    ax.set_title(title)

    logger.info(f"Saving plot to: {output_path}")
    if output_path is not None:
        try:
            plt.savefig(output_path)
        except Exception as e:
            logger.exception("Unable to save plot: " + str(e))
    if show:
        plt.show()


def plot_bmp_grid(
    data, labels, predictions=None, columns=6, height=8, width=8, show=False, output_path=None
):
    # Sstup subplots dimensions
    bmps = len(data)
    cols = columns
    rows = int(bmps / cols)
    if bmps % cols > 0:
        rows += 1
    
    fig, axs = plt.subplots(rows, cols, figsize=(height, width))
    fig.tight_layout(pad=0.1)

    for i, ax in enumerate(axs.flat):
        if i >= bmps:
            break
        # ind = i+j
        ax.imshow(data[i], cmap="gray")
        # Display the label above the image
        if predictions is None:
            my_title = f"Label: {labels[i]}"
            my_color = "black"
        else:
            my_title = f"L: {labels[i]} - P: {predictions[i]}"
            if labels[i] == predictions[i]:
                my_color = "green"
            else:
                my_color = "red"
        ax.set_title(my_title, color=my_color)
        
        ax.set_axis_off()
    fig.set_figheight(height)
    fig.set_figwidth(width)

    logger.info(f"Saving plot to: {output_path}")
    if output_path is not None:
        try:
            plt.savefig(output_path)
        except Exception as e:
            logger.exception("Unable to save plot: " + str(e))
    if show:
        plt.show()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.success("Nothing to do here....")
    # -----------------------------------------


if __name__ == "__main__":
    app()
