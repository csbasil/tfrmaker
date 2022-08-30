"""Dynamic resizing."""
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
TFR_DIR = "tfrecords/chess/"

# fetch all tfrecords in the directory to a list
tfr_paths = [
    os.path.join(TFR_DIR, file)
    for file in os.listdir(TFR_DIR)
    if os.fsdecode(file).endswith(".tfrecord")
]

# load one or more tfrecords as an iterator object.
dataset = images.load(tfr_paths, batch_size=32)

# iterate one batch and visualize it along with labels.
# default size is of each images in the dataset is (224, 224, 3)
databatch = next(iter(dataset))
display.batch(databatch, LABELS)

# load one or more tfrecords as an iterator object.
dataset_resized = images.load(tfr_paths, batch_size=32, image_size=[32, 32])

# Now each images in the dataset is resized to (32, 32, 3) on the fly
databatch = next(iter(dataset_resized))
display.batch(databatch, LABELS)
