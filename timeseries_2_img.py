import glob
import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from progress_bar import updateProgress


def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))


def cos_sum(a, b):
    """To work with tabulate."""
    return math.cos(a + b)


def gramian_angular_field(series):
    """Compute the Gramian Angular Field of an image"""
    # Min-Max scaling
    min_ = np.amin(series)
    max_ = np.amax(series)
    scaled_series = (2 * series - max_ - min_) / (max_ - min_)

    # Floating point inaccuracy!
    scaled_series = np.where(scaled_series >= 1., 1., scaled_series)
    scaled_series = np.where(scaled_series <= -1., -1., scaled_series)

    # Polar encoding
    phi = np.arccos(scaled_series)
    # Note! The computation of r is not necessary
    r = np.linspace(0, 1, len(scaled_series))

    # GAF Computation (every term of the matrix)
    gaf = tabulate(phi, phi, cos_sum)

    return gaf, phi, r, scaled_series


# making sure writing directories exist
if not os.path.exists(os.path.join('result_images', 'polar')):
    os.makedirs(os.path.join('result_images', 'polar'))
if not os.path.exists(os.path.join('result_images', 'GAF')):
    os.makedirs(os.path.join('result_images', 'GAF'))

# setting read/write directory locations
__file_location__ = os.path.join(os.getcwd(), 'augmented_data', 'Stocks')
__label_location__ = os.path.join(os.getcwd(), 'augmented_data', 'Labels')
__write_location_ = os.path.join(os.getcwd(), 'result_images')
pattern = '.txt'

# Plot's specifics
font = {
    'family': 'serif',
    'color': 'darkblue',
    'weight': 'normal',
    'size': 16,
}

# for each file in folder
for path, subdirs, files in os.walk(__file_location__):
    # to keep track of what's been worked on
    total_files = len(files)
    i = 0
    name = ''

    for fname in files:
        name = fname
        if fname.endswith(pattern):
            # Update the progress bar
            progress = float(i / total_files), (i + 1)
            updateProgress(progress[0], progress[1], total_files, os.path.basename(fname))

            # check that file is not empty
            if os.stat(os.path.join(path, fname)).st_size != 0:
                # reading the source csv
                df = pd.read_csv(os.path.join(path, fname), header=0, parse_dates=[0], index_col=[0])

                # only want the close for this test
                df_close = df.drop(['Open', 'High', 'Low', 'Volume', 'OpenInt', 'Action'], axis=1)

                # Get data for plotting
                gaf, phi, r, scaled_time_serie = gramian_angular_field(df_close)

                # Clear plot
                plt.gcf().clear()

                # Polar encoding
                polar = plt.subplot(111, projection='polar')
                polar.plot(phi, r)
                # polar.set_title("Polar Encoding", fontdict=font)
                polar.set_rticks([0, 1])
                polar.set_rlabel_position(-22.5)
                polar.grid(True)

                # SAVE RESULTS
                if not os.path.exists(os.path.join('result_images', 'polar', os.path.basename(os.path.normpath(path)))):
                    os.makedirs(os.path.join('result_images', 'polar', os.path.basename(os.path.normpath(path))))
                plt.savefig(
                    os.path.join(os.getcwd(), 'result_images', 'polar', os.path.basename(os.path.normpath(path)),
                                 fname.replace('txt', 'png')),
                    bbox_inches='tight')

                # Clear plot
                plt.gcf().clear()

                # Gramian Angular Field
                gaf_plot = plt.subplot(111)
                gaf_plot.matshow(gaf)
                # gaf_plot.set_title("Gramian Angular Field", fontdict=font)
                gaf_plot.set_yticklabels([])
                gaf_plot.set_xticklabels([])

                # SAVE RESULTS
                if not os.path.exists(os.path.join('result_images', 'GAF', os.path.basename(os.path.normpath(path)))):
                    os.makedirs(os.path.join('result_images', 'GAF', os.path.basename(os.path.normpath(path))))
                plt.savefig(
                    os.path.join(os.getcwd(), 'result_images', 'GAF', os.path.basename(os.path.normpath(path)),
                                 fname.replace('txt', 'png')),
                    bbox_inches='tight')

            i = i + 1
    updateProgress(1, total_files, total_files, os.path.basename(name))

# TODO : need a progress bar for folder
