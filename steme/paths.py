import os

PROJECT_FOLDER = os.getcwd()
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
DATASET_FOLDER = os.path.join(os.environ["HOME"], "datasets")
FIG_FOLDER = os.path.join(PROJECT_FOLDER, "figures")
LOG_FOLDER = os.path.join(PROJECT_FOLDER, "../logs")
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "models")
# DATASET_FOLDER = "/scratch/mf3734/share/datasets"
DATASET_FOLDER = "/scratch/gv2167/datasets"
PRE_COMPUTED_DATA_FOLDER = os.path.join(DATA_FOLDER, "pre_computed_tempograms")
