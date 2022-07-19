"""Common fixtures for pytest."""
import os
import pytest
import numpy
from PIL import Image


def get_rgb_image(img_dim, img_type="jpg"):
    """Create test rgb image."""
    imarray = numpy.random.rand(*img_dim) * 255
    if img_type == "png":
        return Image.fromarray(imarray.astype("uint8")).convert("RGBA")

    return Image.fromarray(imarray.astype("uint8")).convert("RGB")


@pytest.fixture(scope="class")
def test_temp_dir(tmp_path_factory):
    """Create temporary directory."""
    yield tmp_path_factory.mktemp("test_dir", numbered=False)


@pytest.fixture(scope="class")
def test_temp_dir_basename(tmp_path_factory):
    """Create or get temporary directory basename."""
    yield tmp_path_factory.getbasetemp()


@pytest.fixture(scope="class")
def test_create_image_dataset(
    test_temp_dir, request
):  # pylint: disable=redefined-outer-name
    """Create test image dataset with directory names as class lables."""
    print(request.param)
    image_dir = os.path.join(test_temp_dir, request.param["name"])
    for label in request.param["labels"]:
        label_dir = os.path.join(image_dir, label)
        os.makedirs(label_dir)
        print(label_dir)
        for i in range(0, request.param["size_per_class"]):
            img = get_rgb_image(request.param["dimensions"], request.param["type"])
            img_name = label_dir + f"/{i}.{request.param['type']}"
            img.save(img_name)
    yield request.param["create_options"], request.param["name"], request.param[
        "labels"
    ], image_dir
