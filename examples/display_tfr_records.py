"""Load and display tfrecords."""
import os
from tfrmaker import images, display

# pylint: disable=R0801

# mapping label names with integer encoding.
LABELS = {
    "bishop": 0,
    "knight": 1,
    "pawn": 2,
    "queen": 3,
    "rook": 4,
}

# directory contains tfrecords
TFR_DIR = "examples/tfrecords/chess/"

# fetch all tfrecords in the directory to a list
tfr_paths = [
    os.path.join(TFR_DIR, file)
    for file in os.listdir(TFR_DIR)
    if os.fsdecode(file).endswith(".tfrecord")
]

# load one or more tfrecords as an iterator object.
dataset = images.load(tfr_paths, batch_size=32)

# iterate one batch and visualize it along with labels.
databatch = next(iter(dataset))
display.batch(databatch, LABELS)
