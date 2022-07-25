"""Helper functions to support tfrmaker."""
import os
from typing import Optional, Union, Tuple
from enum import Enum
import tensorflow as tf


def _int64_feature(value: Union[bool, Enum, int]) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value: Union[float]) -> tf.train.Feature:
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value: Union[str]) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_output_dir(directory: str) -> str:
    """Check directory exists or create new one."""

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    return directory


def _split_data_set(
    len_images: int,
    train_split: Optional[float] = None,
    val_split: Optional[float] = None,
) -> Tuple:
    """Split data set into training and validation sets."""

    len_train, len_val = None, None

    len_train = int(len_images * train_split) if train_split else None

    if len_train:
        len_val = int(len_train * val_split) if val_split else None

    return len_train, len_val


def _get_dir_size(path: str = ".") -> tuple[float, float]:
    """Get directory size."""

    size = 0
    max_file_size = 0
    with os.scandir(path) as path_iter:
        for entry in path_iter:
            if entry.is_file():
                f_size = entry.stat().st_size
                max_file_size = f_size if max_file_size < f_size else max_file_size
                size += f_size
    return float(size / 1024 / 1024), float(max_file_size / 1024 / 1024)


def _get_optimal_shards(
    path: str, shard_size: int = 10, host_no: int = 1
) -> tuple[int, int]:
    """Get optimal number of shards and files per shard."""

    size, max_file_size = _get_dir_size(path)
    optimal_shards = -(size // -shard_size * host_no) if int(size) > 0 else 1
    return int(optimal_shards), int(shard_size // max_file_size)
