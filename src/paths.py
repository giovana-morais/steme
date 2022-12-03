import os

PROJECT_FOLDER = os.getcwd()
LOG_FOLDER = os.path.join(PROJECT_FOLDER, "../logs")
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "models")
DATASET_FOLDER = "/scratch/mf3734/share/datasets"
PRE_COMPUTED_DATA_FOLDER = os.path.join(DATA_FOLDER, "pre_computed_tempograms")
