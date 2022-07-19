# tfrmaker
![GitHub](https://img.shields.io/github/license/csbasil/tfrmaker?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/csbasil/tfrmaker?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues-raw/csbasil/tfrmaker?color=yellow&style=flat-square)
> Utility package which helps to ease the manipulation of tfrecords.

## [Contents]()
- [Description](#description)
- [Installation](#instalation)
- [Usage](#usage)
- [Support](#support)
- [Contribute](#contribute)
- [License](#license)

## [Description](description)
[tfrmaker](/) helps to ease the manipulation of tfrecords for your next machine learning project with [tensorflow](https://github.com/tensorflow/tensorflow). You can now easily create, extract and load image datasets in the form of tfrecords with help of [tfrmaker](/). With the help of the package, large image datasets can be converted into tfrecords and fed directly into tensorflow models for training and testing purposes. Some key feature of the package includes:
- dynamic resizing
- spliting training and testing of tfrecords
- spliting training, validation, testing of tfrecords

### Why TFRecords?
TFRecords stores data as a sequence of binary records with help of [protocol buffers](https://developers.google.com/protocol-buffers/), a cross-platform, cross-language library. It has many advantages like:

- **Efficient storage**: TFRecord data can take up less space than the original data; it can also be partitioned into multiple files.
- **Fast I/O**: TFRecord format can be read with parallel I/O operations, which is useful for TPUs or multiple hosts.

## [Installation](instalation)
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install tfrmaker.

```sh
pip install tfrmaker
```

## [Usage](usage)
A minimal usage of `tfrmaker` with image data, organized as directores with name as class labels:
```python
from tfrmaker import images, display

# mapping label names with integer encoding.
LABELS = {"bishop": 0, "knight": 1, "pawn": 2, "queen": 3, "rook": 4}

# specifiying data and output directories.
DATA_DIR = "datasets/chess/"
OUTPUT_DIR = "tfrecords/chess/"

# create tfrecords from the images present in the given data directory.
images.create(DATA_DIR, LABELS, OUTPUT_DIR)

# load one or more tfrecords as an iterator object.
dataset = images.load(["tfrecords/chess/queen.tfrecord","tfrecords/chess/bishop.tfrecords"], batch_size=32, repeat=True)

# iterate one batch and visualize it along with labels.
databatch = next(iter(dataset))
display.batch(databatch, LABELS)
```
Refer [examples](/examples) folder for more advanced usage.

## [Support](support)
> "Your mental support by staring the repo is much appreciated."

## [Contribute](contribute)
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## [License](license)
[MIT](https://choosealicense.com/licenses/mit/)
