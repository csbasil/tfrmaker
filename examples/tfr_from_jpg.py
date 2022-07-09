"""from tfrmaker import tfrm, display

LABELS = {"bishop": 0, "knight": 1, "pawn": 2, "queen": 3, "rook": 4}
DATA_DIR = "../../examples/datasets/chess/"
OUTPUT_DIR = "../../examples/tfrecords/chess"

# create_tf_records(DATA_DIR, LABELS, OUTPUT_DIR)
data = load(
    ["../../examples/tfrecords/chess/bishop.tfrecord"], batch_size=32, repeat=True
)

databatch = next(iter(data))
batch(databatch, LABELS)
"""
