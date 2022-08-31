"""Create TF records from an image directory with class names as sub directories."""
from tfrmaker import images

# mapping label names with integer encoding.
LABELS = {"bishop": 0, "knight": 1, "pawn": 2, "queen": 3, "rook": 4}

# specifiying data and output directories.
DATA_DIR = "examples/datasets/chess/"
OUTPUT_DIR = "examples/tfrecords/chess/"

# create tfrecords from the images present in the given data directory.
info = images.create(DATA_DIR, LABELS, OUTPUT_DIR)

# info contains a list of information
# (path: releative path, size: no of images in the tfrecord)
# about created tfrecords
print(info)
