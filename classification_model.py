import h5py
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model
from keras.applications import InceptionV3, ResNet50, Xception
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

__read_location__ = os.path.realpath(os.path.join(os.getcwd(), 'pickled_dataset'))
__write_location__ = os.path.realpath(os.path.join(os.getcwd(), 'model'))

# network and training
EPOCHS = 200
BATCH_SIZE = 32
VERBOSE = 1
# https://keras.io/optimizers
OPTIMIZER = Adam(lr=0.001, amsgrad=True)

def readFile(stock_name, part_type, X_img=None, Y_next_day=None):
    print("Reading", stock_name, part_type, "data...")
    file_name = stock_name + "_" + part_type + ".hdf5"
    with h5py.File(os.path.join(__read_location__, stock_name, file_name), "r+") as f:
        f_img = f["img"][()]
        f_next_day = f["action"][()]
        f.close()
    if X_img is None:
        X_img = f_img
    else:
        X_img = np.concatenate((X_img, f_img), axis=0)

    if Y_next_day is None:
        Y_next_day = f_next_day
    else:
        Y_next_day = np.concatenate((Y_next_day, f_next_day), axis=0)

    return X_img, Y_next_day


# Load data
print("...loading data")
for path, subdirs, files in os.walk(__read_location__):
    if len(subdirs) == 0:
        stock_name = os.path.basename(os.path.normpath(path))
        img_train, action_train = readFile(stock_name, "train")
        img_valid, action_valid = readFile(stock_name, "validation")
        img_test, action_test = readFile(stock_name, "test")

        print("img_train shape:", img_train.shape)
        print("close_train shape:", action_train.shape)
        print("img_valid shape:", img_valid.shape)
        print("close_valid shape:", action_valid.shape)
        print("img_test shape:", img_test.shape)
        print("v shape:", action_test.shape)

        # encode class values as integers (one hot encoded)
        print("encoding labels...")
        encoder = LabelEncoder()

        # encoding train set
        encoder.fit(action_train)
        encoded_Y_train = encoder.transform(action_train)
        action_train_encoded = keras.utils.np_utils.to_categorical(encoded_Y_train)
        # encoding valid set
        encoder.fit(action_valid)
        encoded_Y_valid = encoder.transform(action_valid)
        action_valid_encoded = keras.utils.np_utils.to_categorical(encoded_Y_valid)
        # encoding test set
        encoder.fit(action_test)
        encoded_Y_test = encoder.transform(action_test)
        action_test_encoded = keras.utils.np_utils.to_categorical(encoded_Y_test)

        # input layer
        image_input = Input(shape=img_train.shape[1:], name="image_input")

        # Image processing layer
        base_cnn_model = ResNet50(weights="imagenet")
        # ResNet50 output from input layer
        x = base_cnn_model(image_input)

        # We stack dense layers and dropout layers to avoid overfitting after that
        x = Dense(1000, activation="relu")(x)
        x = Dropout(0.4)(x)
        x = Dense(1000, activation="relu")(x)
        x = Dropout(0.2)(x)

        # Predictions
        predictions = Dense(3, activation='softmax')(x)

        model = Model(inputs=[image_input], outputs=predictions)

        # printing a model summary to check what we constructed
        print(model.summary())

        # compiling model
        print("compiling model...")
        model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

        # Callbacks
        # Reduce learning rate
        reduceLROnPlat = ReduceLROnPlateau(
            monitor="val_loss", factor=0.8, patience=5, verbose=1, min_lr=0.0001
        )

        # Path to save model
        PATH_SAVE_MODEL = os.path.join(__write_location__, "model_backup")

        # Save weights after every epoch
        if not os.path.exists(PATH_SAVE_MODEL):
            os.makedirs(PATH_SAVE_MODEL)

        csv_logger = CSVLogger(os.path.join(PATH_SAVE_MODEL, "training.csv"))

        print("training ...")
        history = model.fit(
            [img_train],
            [action_train_encoded],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=VERBOSE,
            validation_data=([img_valid], [action_valid_encoded]),
            callbacks=[reduceLROnPlat, csv_logger],
        )

        # serialize weights to HDF5
        model.save_weights(os.path.join(PATH_SAVE_MODEL, "model.h5"))
        print("Saved model to disk")

        score = model.evaluate([img_test], action_test_encoded, batch_size=BATCH_SIZE, verbose=VERBOSE)

        print("Test accuracy:", score[2])

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

        # Reduce learning rate
        plt.plot(history.history["lr"], label="Reduce learning rate")
        plt.title("Reduce learning rate")
        plt.xlabel("Epoch")
        plt.ylabel("Reduce learning rate")
        plt.legend(loc="upper left")
        plt.show()
