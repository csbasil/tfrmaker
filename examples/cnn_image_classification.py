"""CNN image classsifaction with tfrmaker."""
import os
from keras import layers, models
from keras.applications.vgg16 import VGG16
from tfrmaker import images

# mapping label names with integer encoding.
LABELS = {"bishop": 0, "knight": 1, "pawn": 2, "queen": 3, "rook": 4}

# specifiying data and output directories.
DATA_DIR = "datasets/chess/"
OUTPUT_DIR = "tfrecords/chess/"


def get_tfr_paths(tfr_dir):
    """Get list of all tfrecords in the given path."""

    return [
        os.path.join(tfr_dir, file)
        for file in os.listdir(tfr_dir)
        if os.fsdecode(file).endswith(".tfrecord")
    ]


def get_feature(dataset):
    """Filter image_raw and label features from the dataset.
    Original dataset contains: height, width, depth, label, image_raw."""
    return dataset["image_raw"], dataset["label"]


def get_model():
    """Create a model with transfer learning."""

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    for layer in base_model.layers:
        layer.trainable = False

    base_x = base_model.output
    base_x = layers.Flatten()(base_x)
    base_x = layers.Dense(500, activation="relu")(base_x)
    base_x = layers.Dense(500, activation="relu")(base_x)
    predictions = layers.Dense(5, activation="softmax")(base_x)
    return models.Model(inputs=base_model.input, outputs=predictions)


# create tfrecords
info = images.create(DATA_DIR, LABELS, OUTPUT_DIR, train_split=0.8, val_split=0.2)

TRAIN_DIR = "tfrecords/chess/train/"
VAL_DIR = "tfrecords/chess/val/"
TEST_DIR = "tfrecords/chess/test/"

# training dataset from tfrecords
train_dataset = images.load(
    get_tfr_paths(TRAIN_DIR),
    image_size=[32, 32],
    batch_size=32,
    shuffle=True,
    prefetch=True,
    repeat=False,
)

# validation dataset from tfrecords
val_dataset = images.load(
    get_tfr_paths(VAL_DIR),
    image_size=[32, 32],
    batch_size=32,
    shuffle=True,
    prefetch=True,
    repeat=False,
)

# testing dataset from tfrecords
test_dataset = images.load(
    get_tfr_paths(TEST_DIR),
    image_size=[32, 32],
    batch_size=32,
    shuffle=True,
    prefetch=True,
    repeat=False,
)

# compile and train the model
model = get_model()

print(model.summary())

model.compile(
    loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
)

model.fit(
    train_dataset.map(get_feature),
    validation_data=val_dataset.map(get_feature),
    epochs=30,
)

_, accuracy = model.evaluate(test_dataset.map(get_feature))
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
