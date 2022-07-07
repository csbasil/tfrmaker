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


def create_image_example(image_string: str, label_value: int) -> tf.train.Example:
    """Create tensorflow example."""

    image_shape = tf.io.decode_image(image_string).shape
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
    image_size: Optional[List[int]] = None,
) -> None:
    """Write image features into TFRecords."""

    if image_size is None:
        image_size = [32, 32]

    with tf.io.TFRecordWriter(tfrecord_file_name) as writer:

        for image in images:
            image_path = data_dir + label_name + "/" + image
            print(image_path)
            example = create_image_example(
                tf.image.resize(tf.io.read_file(image_path), [*image_size, 3]),
                label_value,
            )
            writer.write(example.SerializeToString())


def create_tf_records(
    data_dir: str,
    labels: Dict[str, int],
    output_dir: str,
    image_size: Optional[List[int]] = None,
    train_split: Optional[float] = None,
    val_split: Optional[float] = None,
):
    """Create TFRecords from the image directory."""

    create_output_dir(output_dir)

    for label_name, label_value in labels.items():
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
                image_size,
            )

            create_output_dir_train(output_dir)
            tfrecord_file_name = output_dir + "train/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir,
                data_path[len_val:len_train],
                tfrecord_file_name,
                label_name,
                label_value,
                image_size,
            )

            create_output_dir_test(output_dir)
            tfrecord_file_name = output_dir + "test/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir,
                data_path[len_train:],
                tfrecord_file_name,
                label_name,
                label_value,
                image_size,
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
                image_size,
            )

            create_output_dir_test(output_dir)
            tfrecord_file_name = output_dir + "test/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir,
                data_path[len_train:],
                tfrecord_file_name,
                label_name,
                label_value,
                image_size,
            )

        else:
            tfrecord_file_name = output_dir + "/" + label_name + ".tfrecord"
            tf_record_writer(
                data_dir,
                data_path,
                tfrecord_file_name,
                label_name,
                label_value,
                image_size,
            )
