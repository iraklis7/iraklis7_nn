import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATASET = os.environ["IRAKLIS7_NN_DATASET"]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_SET = DATASET + "_raw"
TRAIN_SET = DATASET + "_train"
VAL_SET = DATASET + "_eval"
TEST_SET = DATASET + "_test"
PRED_SET = DATASET + "_predictions"

MODEL_NAME = DATASET + ".keras"

SAMPLE_PLOT = DATASET + "_ds_sample.png"
SAMPLE_MC_PLOT = DATASET + "_ds_sample_mc.png"
MCS_PER_LABEL_PLOT = DATASET + "_mcs_per_label.png"
DIST_MCL_PLOT = DATASET + "_dist_mcl.png"
DIST_PLOT = DATASET + "_dist_for_"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
