"""Test tfrmaker images module."""
import os
import pytest
from .context import src  # pylint: disable=unused-import
from src.tfrmaker import images  # pylint: disable=wrong-import-order
from .utils_test import inject_test_data

test_data = inject_test_data("tests/data_test.json")


class TestImages:
    """Testing basic operations of images module."""

    @pytest.mark.parametrize(
        "test_create_image_dataset",
        test_data["image_datasets_from_dir"],
        indirect=["test_create_image_dataset"],
    )
    def test_create(self, test_temp_dir, test_create_image_dataset):
        """Test tfrecord creation from image directory."""

        create_options, name, labels, data_dir = test_create_image_dataset
        output_dir = str(test_temp_dir) + "/tfrecords/" + name + "/"

        images.create(str(data_dir) + "/", labels, output_dir, **create_options)

        for label in labels:
            if "train_split" in create_options and "val_split" in create_options:
                print(os.listdir(output_dir))
                assert os.path.exists(output_dir + "val/" + label + ".tfrecord") is True
                assert (
                    os.path.exists(output_dir + "train/" + label + ".tfrecord") is True
                )
                assert (
                    os.path.exists(output_dir + "test/" + label + ".tfrecord") is True
                )

            elif "train_split" in create_options:
                print(os.listdir(output_dir))
                assert (
                    os.path.exists(output_dir + "train/" + label + ".tfrecord") is True
                )
                assert (
                    os.path.exists(output_dir + "test/" + label + ".tfrecord") is True
                )
            else:
                print(os.listdir(output_dir))
                assert os.path.exists(output_dir + label + ".tfrecord") is True

    @pytest.mark.parametrize("image_datasets", test_data["image_datasets_from_dir"])
    def test_load(self, test_temp_dir_basename, image_datasets):
        """Test tfrecord loading as batches."""

        if "val_split" in image_datasets["create_options"]:
            tfrecord_paths_val = (
                f"{test_temp_dir_basename}"
                f"/test_dir/tfrecords/{image_datasets['name']}/val"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths_val, file)
                for file in os.listdir(tfrecord_paths_val)
                if file.endswith(".tfrecord")
            ]
            val_dataset = images.load(tfrecord_names, **image_datasets["load_options"])
            data = list((val_dataset.take(1)).as_numpy_iterator())
            assert "width" in data[0]
            assert "height" in data[0]
            assert "depth" in data[0]
            assert "image_raw" in data[0]

        if "train_split" in image_datasets["create_options"]:
            tfrecord_paths_train = (
                f"{test_temp_dir_basename}"
                f"/test_dir/tfrecords/{image_datasets['name']}/train"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths_train, file)
                for file in os.listdir(tfrecord_paths_train)
                if file.endswith(".tfrecord")
            ]
            train_dataset = images.load(
                tfrecord_names, **image_datasets["load_options"]
            )
            data = list((train_dataset.take(1)).as_numpy_iterator())
            assert "width" in data[0]
            assert "height" in data[0]
            assert "depth" in data[0]
            assert "image_raw" in data[0]

            tfrecord_paths_test = (
                f"{test_temp_dir_basename}"
                f"/test_dir/tfrecords/{image_datasets['name']}/test"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths_test, file)
                for file in os.listdir(tfrecord_paths_test)
                if file.endswith(".tfrecord")
            ]
            test_dataset = images.load(tfrecord_names, **image_datasets["load_options"])
            data = list((test_dataset.take(1)).as_numpy_iterator())
            assert "width" in data[0]
            assert "height" in data[0]
            assert "depth" in data[0]
            assert "image_raw" in data[0]

        else:
            tfrecord_paths = (
                f"{test_temp_dir_basename}/test_dir/tfrecords/{image_datasets['name']}"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths, file)
                for file in os.listdir(tfrecord_paths)
                if file.endswith(".tfrecord")
            ]
            dataset = images.load(tfrecord_names, **image_datasets["load_options"])
            data = list((dataset.take(1)).as_numpy_iterator())
            assert "width" in data[0]
            assert "height" in data[0]
            assert "depth" in data[0]
            assert "image_raw" in data[0]

    @pytest.mark.parametrize("image_datasets", test_data["image_datasets_from_dir"])
    def test_count(self, test_temp_dir_basename, image_datasets):
        """Test tfrecord counting."""
        size_per_class = image_datasets["size_per_class"]
        if (
            "train_split" in image_datasets["create_options"]
            and "val_split" in image_datasets["create_options"]
        ):
            train_size = int(
                size_per_class
                * len(image_datasets["labels"])
                * image_datasets["create_options"]["train_split"]
            )
            val_size = int(train_size * image_datasets["create_options"]["val_split"])
            tfrecord_paths_val = (
                f"{test_temp_dir_basename}"
                f"/test_dir/tfrecords/{image_datasets['name']}/val"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths_val, file)
                for file in os.listdir(tfrecord_paths_val)
                if file.endswith(".tfrecord")
            ]
            count = images.count(tfrecord_names)
            assert count == val_size

            tfrecord_paths_train = (
                f"{test_temp_dir_basename}"
                f"/test_dir/tfrecords/{image_datasets['name']}/train"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths_train, file)
                for file in os.listdir(tfrecord_paths_train)
                if file.endswith(".tfrecord")
            ]
            count = images.count(tfrecord_names)
            assert count == (train_size - val_size)

            test_size = size_per_class * len(image_datasets["labels"]) - train_size
            tfrecord_paths_test = (
                f"{test_temp_dir_basename}"
                f"/test_dir/tfrecords/{image_datasets['name']}/test"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths_test, file)
                for file in os.listdir(tfrecord_paths_test)
                if file.endswith(".tfrecord")
            ]
            count = images.count(tfrecord_names)
            assert count == test_size

        elif "train_split" in image_datasets["create_options"]:
            train_size = int(
                size_per_class
                * len(image_datasets["labels"])
                * image_datasets["create_options"]["train_split"]
            )
            tfrecord_paths_train = (
                f"{test_temp_dir_basename}"
                f"/test_dir/tfrecords/{image_datasets['name']}/train"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths_train, file)
                for file in os.listdir(tfrecord_paths_train)
                if file.endswith(".tfrecord")
            ]
            count = images.count(tfrecord_names)
            assert count == train_size

            test_size = size_per_class * len(image_datasets["labels"]) - train_size
            tfrecord_paths_test = (
                f"{test_temp_dir_basename}"
                f"/test_dir/tfrecords/{image_datasets['name']}/test"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths_test, file)
                for file in os.listdir(tfrecord_paths_test)
                if file.endswith(".tfrecord")
            ]
            count = images.count(tfrecord_names)
            assert count == test_size

        else:
            tfrecord_paths = (
                f"{test_temp_dir_basename}/test_dir/tfrecords/{image_datasets['name']}"
            )
            tfrecord_names = [
                os.path.join(tfrecord_paths, file)
                for file in os.listdir(tfrecord_paths)
                if file.endswith(".tfrecord")
            ]
            print(tfrecord_names)
            count = images.count(tfrecord_names)
            size_per_class = size_per_class * len(image_datasets["labels"])
            assert count == size_per_class
