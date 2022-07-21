"""Utility functions to create and load TFRecords from images."""

import os
from typing import List, Optional, Dict
import asyncio

import tensorflow as tf

from .helper import (
    _int64_feature,
    _bytes_feature,
    _create_output_dir,
    _split_data_set,
)

AUTOTUNE = tf.data.AUTOTUNE

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False


def _decode_image(image: str, size: List[int] = None):
    """Decode, resize, and normalize raw images."""

    image = tf.image.decode_image(image, expand_animations=False)
    if size:
        image = tf.image.resize(image, size=size)
    image = tf.cast(image, tf.float64) / 255.0
    return image


def _create_image_example(image_string: str, label_value: int) -> tf.train.Example:
    """Create tensorflow example with image features."""

    decoded_image = tf.image.decode_image(image_string, expand_animations=False)
    image_shape = decoded_image.shape

    feature = {
        "height": _int64_feature(image_shape[0]),
        "width": _int64_feature(image_shape[1]),
        "depth": _int64_feature(image_shape[2]),
        "label": _int64_feature(label_value),
        "image_raw": _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _tf_record_image_writer(
    data_dir: str,
    images: List[str],
    tfrecord_file_name: str,
    label_name: str,
    label_value: int,
):
    """Write image features into TFRecords."""

    with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
        for image in images:
            image_path = data_dir + label_name + "/" + image
            image_string = tf.io.read_file(image_path)
            example = _create_image_example(image_string, label_value)
            writer.write(example.SerializeToString())


async def _create_with_train_val_split(
    data_dir: str,
    output_dir: str,
    label_name: str,
    label_value: int,
    len_train: int,
    len_val: int,
):
    data_path = os.listdir(data_dir + label_name + "/")
    tfrecord_val_file_name = (
        _create_output_dir(output_dir + "val/") + label_name + ".tfrecord"
    )
    await asyncio.gather(
        _create_with_train_split(
            data_dir, output_dir, label_name, label_value, len_train, len_val
        ),
        asyncio.get_running_loop().run_in_executor(
            None,
            _tf_record_image_writer,
            data_dir,
            data_path[0:len_val],
            tfrecord_val_file_name,
            label_name,
            label_value,
        ),
    )


async def _create_with_train_split(
    data_dir: str,
    output_dir: str,
    label_name: str,
    label_value: int,
    len_train: int,
    len_val: int = 0,
):
    data_path = os.listdir(data_dir + label_name + "/")
    tfrecord_train_file_name = (
        _create_output_dir(output_dir + "train/") + label_name + ".tfrecord"
    )
    tfrecord_test_file_name = (
        _create_output_dir(output_dir + "test/") + label_name + ".tfrecord"
    )

    tasks: list = []

    tasks.append(
        asyncio.get_running_loop().run_in_executor(
            None,
            _tf_record_image_writer,
            data_dir,
            data_path[len_val:len_train],
            tfrecord_train_file_name,
            label_name,
            label_value,
        )
    )

    tasks.append(
        asyncio.get_running_loop().run_in_executor(
            None,
            _tf_record_image_writer,
            data_dir,
            data_path[len_train:],
            tfrecord_test_file_name,
            label_name,
            label_value,
        )
    )

    await asyncio.gather(*tasks)


async def _create_from_dir(
    data_dir: str,
    label_mappings: Dict[str, int],
    output_dir: str,
    train_split: Optional[float] = None,
    val_split: Optional[float] = None,
):
    _create_output_dir(output_dir)

    tasks: list = []

    for label_name, label_value in label_mappings.items():
        data_path = os.listdir(data_dir + label_name + "/")

        len_train, len_val = _split_data_set(
            int(len(data_path)), train_split, val_split
        )

        if train_split and val_split:
            tasks.append(
                _create_with_train_val_split(
                    data_dir, output_dir, label_name, label_value, len_train, len_val
                )
            )

        elif train_split:
            tasks.append(
                _create_with_train_split(
                    data_dir, output_dir, label_name, label_value, len_train
                )
            )

        else:
            tfrecord_file_name = (
                _create_output_dir(output_dir) + label_name + ".tfrecord"
            )
            tasks.append(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    _tf_record_image_writer,
                    data_dir,
                    data_path,
                    tfrecord_file_name,
                    label_name,
                    label_value,
                )
            )

    await asyncio.gather(*tasks)


def create(
    data_dir: str,
    label_mappings: Dict[str, int],
    output_dir: str,
    method: str = "dir",
    train_split: Optional[float] = None,
    val_split: Optional[float] = None,
):
    """Create TFRecords from the images."""
    if method == "dir":
        asyncio.run(
            _create_from_dir(
                data_dir, label_mappings, output_dir, train_split, val_split
            )
        )


def _extract(tfrecord: str, image_size: List[int] = None):
    """Extract image features from tfrecords."""

    features = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
    }

    # Extract the data record
    example = tf.io.parse_single_example(tfrecord, features)
    example["image_raw"] = _decode_image(example["image_raw"], image_size)
    return example


def load(
    tfrecord_paths: List[str],
    batch_size: int = None,
    image_size: List[int] = None,
    repeat: bool = False,
    shuffle: bool = False,
    prefetch: bool = True,
):
    """Load tfrecord dataset for traning."""

    dataset = tf.data.TFRecordDataset(tfrecord_paths, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        lambda tfrecord: _extract(tfrecord, image_size), num_parallel_calls=AUTOTUNE
    )
    if prefetch:
        dataset = dataset.prefetch(AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(shuffle)
    if batch_size:
        dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset


def count(tfrecord_paths: List[str]) -> int:
    """Count no of examples in a list of tfrecords."""

    dataset = load(tfrecord_paths, batch_size=1)
    return dataset.reduce(0, lambda x, _: x + 1).numpy()
