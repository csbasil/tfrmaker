"""Benchmark tfrmaker."""
import os
from time import time
from functools import wraps
from tfrmaker import images

LABELS = {"bishop": 0, "knight": 1, "pawn": 2, "queen": 3, "rook": 4}

# specifiying data and output directories.
DATA_DIR = "examples/datasets/chess/"
OUTPUT_DIR = "examples/tfrecords/chess/"


def _time_tracker(log_fun):
    def _tracker(inner_fn):
        @wraps(inner_fn)
        def wrapped_fn(*args, **kwargs):
            start_time = time()

            try:
                result = inner_fn(*args, **kwargs)
            finally:
                elapsed_time = time() - start_time

                # log the result
                log_fun(
                    {
                        "function_name": inner_fn.__name__,
                        "total_time": elapsed_time,
                    }
                )

            return result

        return wrapped_fn

    return _tracker


def _log(message):
    """Display the time taken."""
    print(f"{message['function_name']} {message['total_time']:.3f}")


@_time_tracker(_log)
def images_create():
    """Create tfrcords and track time."""
    images.create(DATA_DIR, LABELS, OUTPUT_DIR)


@_time_tracker(_log)
def images_load():
    """Load tfrecords and track time."""
    tfr_paths = [
        os.path.join(OUTPUT_DIR, file)
        for file in os.listdir(OUTPUT_DIR)
        if os.fsdecode(file).endswith(".tfrecord")
    ]
    images.load(
        tfr_paths,
        batch_size=16,
        shuffle=True,
    )


@_time_tracker(_log)
def images_count():
    """Count tfrecords and track time."""
    tfr_paths = [
        os.path.join(OUTPUT_DIR, file)
        for file in os.listdir(OUTPUT_DIR)
        if os.fsdecode(file).endswith(".tfrecord")
    ]
    return images.count(tfr_paths)


@_time_tracker(_log)
def profile_tfr_images():
    """Overall performance tracking."""
    # create tf records
    images_create()

    # load tf records
    images_load()

    # count tf records
    images_count()


profile_tfr_images()
