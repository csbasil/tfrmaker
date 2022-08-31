"""
Count number of images inside a list of tfrecords.
"""
import os
from tfrmaker import images

# directory contains tfrecords
TFR_DIR = "examples/tfrecords/chess/"

# fetch all tfrecords in the directory to a list
tfr_paths = [
    os.path.join(TFR_DIR, file)
    for file in os.listdir(TFR_DIR)
    if os.fsdecode(file).endswith(".tfrecord")
]

# count no of items in the tfrecords
count = images.count(tfr_paths)

print(count)
