"""
Create TF records from an image directory with class names as sub directories
and split them into training and validaton datasets.
"""
from tfrmaker import images

# mapping label names with integer encoding.
LABELS = {"bishop": 0, "knight": 1, "pawn": 2, "queen": 3, "rook": 4}

# specifiying data and output directories.
DATA_DIR = "examples/datasets/chess/"
OUTPUT_DIR = "examples/tfrecords/chess/"

# split 80% of image from each label for training and 20% for testing if train_split=0.8.
# split 20% of training images from each label for validation if val_split=0.2
# Both train_split and val_split ranges from (0,1)
info = images.create(DATA_DIR, LABELS, OUTPUT_DIR, train_split=0.8, val_split=0.2)

# info contains a list of information:
# (path: releative path, size: no of images in the tfrecord)
# about created tfrecords
print(info)
