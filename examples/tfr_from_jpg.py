"""Simple tfrmaker example."""

from tfrmaker import images, display

LABELS = {"bishop": 0, "knight": 1, "pawn": 2, "queen": 3, "rook": 4}
DATA_DIR = "datasets/chess/"
OUTPUT_DIR = "tfrecords/chess"

# tfrm.create(DATA_DIR, LABELS, OUTPUT_DIR)

data = images.load(["tfrecords/chess/queen.tfrecord"], batch_size=32, repeat=True)

databatch = next(iter(data))
display.batch(databatch, LABELS)
