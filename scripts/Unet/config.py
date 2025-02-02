import torch
import os


DATASET_PATH = r"/home/imad/NeedleAlignment/data/needleonly"
TEST_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if DEVICE == "cuda" else False

INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64

THRESHOLD = 0.5

BASE_OUTPUT = r"/home/imad/NeedleAlignment/"

MODEL_PATH = os.path.join(BASE_OUTPUT, "Models", "unet_model.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])