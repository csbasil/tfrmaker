"""Create tfrecords from images organized as directores
with class/label name as directory name.

Following directory structure is expected:
data_directory/
    class_name1/
        img_1.(jpg,png,bmp)
        img_1.(jpg,png,bmp)
    class_name2/
        img_1.(jpg,png,bmp)
        img_1.(jpg,png,bmp)


"""
from tfrmaker import images, display

# mapping label names with integer encoding.
LABELS = {"bishop": 0, "knight": 1, "pawn": 2, "queen": 3, "rook": 4}

# specifiying data and output directories.
DATA_DIR = "datasets/chess/"
OUTPUT_DIR = "tfrecords/chess"

# create tfrecords from the images present in the given data directory.
images.create(DATA_DIR, LABELS, OUTPUT_DIR)

# load one or more tfrecords as an iterator object.
dataset = images.load(
    ["tfrecords/chess/queen.tfrecord", "tfrecords/chess/bishop.tfrecord"],
    batch_size=12,
    shuffle=True,
)

# iterate one batch and visualize it along with labels.
databatch = next(iter(dataset))
display.batch(databatch, LABELS)

# count no of images inside tfrecord
count = images.count(["tfrecords/chess/queen.tfrecord"])
print(count)
