"""Utility functions to create and load TFRecords from images."""

import os
from typing import List, Optional, Dict
import tensorflow as tf

from tfrmaker.helper import (
    _int64_feature,
    _bytes_feature,
    create_output_dir,
    create_output_dir_train,
    create_output_dir_val,
    create_output_dir_test,
    split_data_set,
)

AUTOTUNE = tf.data.AUTOTUNE

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False


def decode_image(image: str, size: List[int] = None):
    """Decode, resize, and normalize raw images."""

    image = tf.image.decode_image(image, expand_animations=False)
    if size:
        image = tf.image.resize(image, size=size)
    image = tf.cast(image, tf.float64) / 255.0
    return image


def create_image_example(image_string: str, label_value: int) -> tf.train.Example:
    """Create tensorflow example."""

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


def tf_record_writer(
    data_dir: str,
    images: List[str],
    tfrecord_file_name: str,
    label_name: str,
    label_value: int,
) -> None:
    """Write image features into TFRecords."""

    with tf.io.TFRecordWriter(tfrecord_file_name) as writer:

        for image in images:
            image_path = data_dir + label_name + "/" + image
            image_string = tf.io.read_file(image_path)
            example = create_image_example(image_string, label_value)
            writer.write(example.SerializeToString())


def create(
    data_dir: str,
    label_mappings: Dict[str, int],
    output_dir: str,
    train_split: Optional[float] = None,
    val_split: Optional[float] = None,
):
    """Create TFRecords from the image directory."""

    create_output_dir(output_dir)

    for label_name, label_value in label_mappings.items():
        data_path = os.listdir(data_dir + label_name + "/")

        len_train, len_val = split_data_set(int(len(data_path)), train_split, val_split)

        if train_split and val_split:

            create_output_dir_val(output_dir)
            tfrecord_file_name = output_dir + "val/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir,
                data_path[0:len_val],
                tfrecord_file_name,
                label_name,
                label_value,
            )

            create_output_dir_train(output_dir)
            tfrecord_file_name = output_dir + "train/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir,
                data_path[len_val:len_train],
                tfrecord_file_name,
                label_name,
                label_value,
            )

            create_output_dir_test(output_dir)
            tfrecord_file_name = output_dir + "test/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir,
                data_path[len_train:],
                tfrecord_file_name,
                label_name,
                label_value,
            )

        elif train_split:
            create_output_dir_train(output_dir)
            tfrecord_file_name = output_dir + "train/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir,
                data_path[0:len_train],
                tfrecord_file_name,
                label_name,
                label_value,
            )

            create_output_dir_test(output_dir)
            tfrecord_file_name = output_dir + "test/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir,
                data_path[len_train:],
                tfrecord_file_name,
                label_name,
                label_value,
            )

        else:
            tfrecord_file_name = output_dir + "/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir, data_path, tfrecord_file_name, label_name, label_value
            )


def extract(tfrecord: str, image_size: List[int] = None):
    """Extract features from tfrecords."""

    features = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
    }

    # Extract the data record
    example = tf.io.parse_single_example(tfrecord, features)
    image = decode_image(example["image_raw"], image_size)
    label = tf.cast(example["label"], tf.int64)
    return image, label


def load(
    tfrecord_names: List[str],
    shuffle: bool = False,
    batch_size: int = 16,
    image_size: List[int] = None,
    repeat: bool = False,
):
    """Load tfrecord dataset for traning."""

    dataset = tf.data.TFRecordDataset(tfrecord_names, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        lambda tfrecord: extract(tfrecord, image_size), num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.prefetch(AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset
