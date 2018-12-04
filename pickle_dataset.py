#!/usr/bin/python3
import math
import os

import cv2
import h5py
import numpy as np
import pandas as pd
from progress_bar import updateProgress
from scipy import misc

# Directory of dataset to use
TRAIN_DIR = "augmented_data"

__read_location_labels__ = os.path.realpath(os.path.join(os.getcwd(), "augmented_data", 'Labels'))
__read_location_image_GAF__ = os.path.realpath(os.path.join(os.getcwd(), "result_images", 'GAF'))
__write_location__ = os.path.realpath(os.path.join(os.getcwd(), "pickled_dataset"))
img_file = ""


def processImage(img_path):
    # Read and resize image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))

    # Convert the image into an 8 bit array
    return np.asarray(img, dtype=np.float32)


def loadDataSet(folder_path):
    # initialize variables
    image_train = []
    action_label_train = []
    close_label_train = []
    index_file_path = os.path.join(__read_location_labels__, os.path.basename(os.path.normpath(folder_path)) + '.txt')

    # load index and labels
    index_file = pd.read_csv(index_file_path, header=0)

    total_file = len(index_file)
    for i in range(total_file):

        # Update the progress bar
        progress = float(i / total_file), (i + 1)
        updateProgress(progress[0], progress[1], total_file, img_file)

        # Get image's path
        img_path = os.path.join(folder_path, index_file.filename[i])
        if os.path.exists(img_path):
            img = processImage(img_path) / 255.
            image_train.append(img)
            action_label_train.append(index_file.action[i])
            close_label_train.append(index_file.close[i])

    updateProgress(1, total_file, total_file, img_file)

    return image_train, action_label_train, close_label_train


# Write hdf5 file
def writeFile(stock_name, part_type, img, action, close):
    # initialization
    print("Saving", stock_name, part_type, "data...")
    stock_write_path = os.path.join(__write_location__, stock_name)

    # making sure stock folder exists
    if not os.path.exists(stock_write_path):
        os.makedirs(stock_write_path)

    # writing down the file
    file_name = stock_name + "_" + part_type + ".hdf5"
    with h5py.File(os.path.join(stock_write_path, file_name), "w") as f:
        f.create_dataset(
            "img",
            data=img,
            dtype=np.float32,
            compression="gzip", compression_opts=5
        )
        f.create_dataset("action", data=action, dtype=np.uint8)
        f.create_dataset("close", data=close, dtype=np.uint8)
        f.close()


# Save dataset
def saveDataSet(folder_path, image_train, action_label_train, close_label_train):
    # initializing process
    print("Dividing the data set...")
    img = np.asarray(image_train)
    action = np.asarray(action_label_train, dtype=np.uint8)
    close = np.asarray(close_label_train, dtype=np.uint8)
    stock_name = os.path.basename(os.path.normpath(folder_path))

    # making sure saving folder exists
    if not os.path.exists(__write_location__):
        os.makedirs(__write_location__)

    # Split dataset
    k = int(len(image_train) / 6)
    writeFile(stock_name, "test", img[:k, :, :, :], action[:k], close[:k]
              )
    writeFile(stock_name,
              "validation",
              img[k: 2 * k, :, :, :],
              action[k: 2 * k],
              close[k: 2 * k],
              )
    writeFile(
        stock_name, "train", img[2 * k:, :, :, :], action[2 * k:], close[2 * k:]
    )


if __name__ == "__main__":
    for path, subdirs, files in os.walk(__read_location_image_GAF__):
        if len(subdirs) == 0:
            image_train, action_label_train, close_label_train = loadDataSet(path)
            saveDataSet(path, image_train, action_label_train, close_label_train)
