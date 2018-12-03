#!/usr/bin/python3
# ./main.py --load_weight ./weight_file.h5

import argparse
import os

import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import InceptionV3, ResNet50, Xception
from keras.layers import Flatten, Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
from six.moves import cPickle

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-lw", "--load_weights", help="Path to the file weights")
args = vars(ap.parse_args())

__read_location__ = os.path.realpath(os.path.join(os.getcwd(), 'pickled_dataset'))
__write_location__ = os.path.realpath(os.path.join(os.getcwd(), 'model'))

# network and training
EPOCHS = 20
BATCH_SIZE = 32
VERBOSE = 1
# https://keras.io/optimizers
OPTIMIZER = Adam(lr=0.001, amsgrad=True)
# OPTIMIZER = RMSprop()
# OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# OPTIMIZER = Adagrad(lr=0.05)

# Image processing layer
# CNN = 'Xception'
# CNN = 'IV3'
CNN = "RN50"


def readFile(stock_name, part_type, X_img=None, Y_close=None):
    print("Reading", stock_name, part_type, "data...")
    file_name = stock_name + "_" + part_type + ".hdf5"
    with h5py.File(os.path.join(__read_location__, stock_name, file_name), "r+") as f:
        f_img = f["img"][()]
        f_close = f["close"][()]
        f.close()
    if X_img is None:
        X_img = f_img
    else:
        X_img = np.concatenate((X_img, f_img), axis=0)

    if Y_close is None:
        Y_close = f_close
    else:
        Y_close = np.concatenate((Y_close, f_close), axis=0)

    return X_img, Y_close


# Load data
print("...loading training data")
for path, subdirs, files in os.walk(__read_location__):
    if len(subdirs) == 0:
        stock_name = os.path.basename(os.path.normpath(path))
        img_train, close_train = readFile(stock_name, "train")
        img_valid, close_valid = readFile(stock_name, "validation")
        img_test, close_test = readFile(stock_name, "test")

        print("img_train shape:", img_train.shape)
        print("close_train shape:", close_train.shape)
        print("img_valid shape:", img_valid.shape)
        print("close_valid shape:", close_valid.shape)
        print("img_test shape:", img_test.shape)
        print("v shape:", close_test.shape)

        # First we need to create a model structure
        # input layer
        image_input = Input(shape=img_train.shape[1:], name="image_input")

        if CNN == "IV3":
            # Inception V3 layer with pre-trained weights from ImageNet
            # base_iv3_model = InceptionV3(include_top=False, weights="imagenet")
            base_iv3_model = InceptionV3(weights="imagenet")
            # Inception V3 output from input layer
            x = base_iv3_model(image_input)
            # flattening it #why?
            # flat_iv3 = Flatten()(output_vgg16)
        elif CNN == "RN50":
            # ResNet50 layer with pre-trained weights from ImageNet
            base_rn50_model = ResNet50(weights="imagenet")
            # ResNet50 output from input layer
            x = base_rn50_model(image_input)
        elif CNN == "Xception":
            # Xception layer with pre-trained weights from ImageNet
            base_xp_model = Xception(weights="imagenet")
            # Xception output from input layer
            x = base_xp_model(image_input)

        # We stack dense layers and dropout layers to avoid overfitting after that
        x = Dense(1000, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(1000, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(240, activation="relu")(x)
        # x = Dropout(0.1)(x)

        # and the final prediction layer as output
        # predictions = Dense(1, activation='sigmoid', name='predictions')(x)
        predictions = Dense(1)(x)

        # Now that we have created a model structure we can define it
        # this defines the model with one input and one output
        model = Model(inputs=[image_input], outputs=predictions)

        # printing a model summary to check what we constructed
        print(model.summary())

        # Load weight
        if args["load_weights"] != None:
            print("Loading weights for", args["load_weights"])
            model.load_weights(args["load_weights"])

        model.compile(optimizer=OPTIMIZER, loss="mean_squared_error", metrics=["MAE"])

        # Save weights after every epoch
        if not os.path.exists(os.path.join(__write_location__, "weights")):
            os.makedirs(os.path.join(__write_location__, "weights"))

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath="weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
            save_weights_only=True,
            period=1,
        )

        # Reduce learning rate
        reduceLROnPlat = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.8, patience=3, verbose=1, min_lr=0.0001
        )

        # TensorBoard
        # how to use: $ tensorboard --logdir path_to_current_dir/Graph
        # Save log for tensorboard
        LOG_DIR_TENSORBOARD = os.path.join(__write_location__, "tensorboard")
        if not os.path.exists(LOG_DIR_TENSORBOARD):
            os.makedirs(LOG_DIR_TENSORBOARD)

        tbCallBack = keras.callbacks.TensorBoard(
            log_dir=LOG_DIR_TENSORBOARD,
            batch_size=BATCH_SIZE,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        print("tensorboard --logdir", LOG_DIR_TENSORBOARD)

        # Path to save model
        PATH_SAVE_MODEL = os.path.join(__write_location__, "model_backup", stock_name)

        # Save weights after every epoch
        if not os.path.exists(PATH_SAVE_MODEL):
            os.makedirs(PATH_SAVE_MODEL)

        csv_logger = keras.callbacks.CSVLogger(os.path.join(PATH_SAVE_MODEL, "training.csv"))

        history = model.fit(
            [img_train],
            [close_train],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=VERBOSE,
            validation_data=([img_valid], [close_valid]),
            callbacks=[tbCallBack, checkpoint, reduceLROnPlat, csv_logger],
        )

        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open(os.path.join(PATH_SAVE_MODEL, "model.yaml"), "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights(os.path.join(PATH_SAVE_MODEL, "model.h5"))
        print("Saved model to disk")

        score = model.evaluate([img_test], close_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

        print("\nTest loss:", score[0])
        print("Test MAE:", score[1])
        # print("Test accuracy:", score[2])

        # Save all data in history
        with open(os.path.join(PATH_SAVE_MODEL, "history.pkl"), "wb") as f:
            cPickle.dump(history.history, f)
        f.close()

        # list all data in history
        print(history.history.keys())

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()

        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(os.path.join(PATH_SAVE_MODEL, "history_loss.png"))
        plt.close()

        plt.plot(history.history["mean_absolute_error"], label="mean")
        plt.plot(history.history["val_mean_absolute_error"], label="val_mean")
        plt.title("Training Absolute Error")
        plt.xlabel("Epoch")
        plt.ylabel("Absolute Error")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(os.path.join(PATH_SAVE_MODEL, "history_mean.png"))
        plt.close()

        # summarize history for loss
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

        # summarize history for mean
        plt.plot(history.history["mean_absolute_error"], label="mean")
        plt.plot(history.history["val_mean_absolute_error"], label="val_mean")
        plt.title("Training Absolute Error")
        plt.xlabel("Epoch")
        plt.ylabel("Absolute Error")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

        # Reduce learning rate
        plt.plot(history.history["lr"], label="Reduce learning rate")
        plt.title("Reduce learning rate")
        plt.xlabel("Epoch")
        plt.ylabel("Reduce learning rate")
        plt.legend(loc="upper left")
        plt.show()
