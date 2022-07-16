"""Tfmaker display utilities."""
import math
from matplotlib import pyplot as plt  # type: ignore


def batch_to_numpy_images_and_labels(data):
    """Convert batch to images and labels."""

    labels, images = data["label"], data["image_raw"]
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object:
        numpy_labels = [None for _ in enumerate(numpy_images)]
    return numpy_images, numpy_labels


def title_from_label_and_target(label, correct_label, label_mappings):
    """Get title from class value."""

    if correct_label is None:
        return label_mappings[label], True
    correct = label == correct_label
    prediction = "OK" if correct else "NO"
    actual = f"{'-->'+label_mappings[correct_label] if not correct else ''}"
    title = f"{label_mappings[label]} [{prediction}{actual}]"
    return title, correct


def image(img, title, subplot, red=False, titlesize=12):
    """Display one image with title."""

    plt.subplot(*subplot)
    plt.axis("off")
    plt.imshow(img)
    if len(title) > 0:
        plt.title(
            title,
            fontsize=int(titlesize) if not red else int(titlesize / 1.2),
            color="red" if red else "black",
            fontdict={"verticalalignment": "center"},
            pad=int(titlesize / 1.5),
        )
    return (subplot[0], subplot[1], subplot[2] + 1)


def batch(databatch, label_mappings, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]

    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # size and spacing
    figsize = 13.0
    spacing = 0.13
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(figsize, figsize / cols * rows))
    else:
        plt.figure(figsize=(figsize / rows * cols, figsize))
    for count, (img, label) in enumerate(
        zip(images[: rows * cols], labels[: rows * cols])
    ):
        title = (
            ""
            if label is None
            else list(label_mappings.keys())[list(label_mappings.values()).index(label)]
        )
        title = title + str(img.shape)
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(
                predictions[count], label, label_mappings
            )
        dynamic_titlesize = figsize * spacing / max(rows, cols) * 40 + 3
        subplot = image(img, title, subplot, not correct, titlesize=dynamic_titlesize)
        plt.tight_layout()
        if label is None and predictions is None:
            plt.subplots_adjust(wspace=0, hspace=0)
        else:
            plt.subplots_adjust(wspace=spacing, hspace=spacing)
    plt.show()
