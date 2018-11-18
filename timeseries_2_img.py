import glob
import math
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


# Show a progress bar
def updateProgress(progress, tick="", total="", status="Loading..."):
    lineLength = 80
    barLength = 23
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0:
        progress = 0
        status = "Waiting...\r"
    if progress >= 1:
        progress = 1
        status = "Completed loading data\r\n"
    block = int(round(barLength * progress))
    line = str("\rImage: {0}/{1} [{2}] {3}% {4}").format(
        tick,
        total,
        str(("#" * block)) + str("." * (barLength - block)),
        round(progress * 100, 1),
        status,
    )
    emptyBlock = lineLength - len(line)
    emptyBlock = " " * emptyBlock if emptyBlock > 0 else ""
    sys.stdout.write(line + emptyBlock)
    sys.stdout.flush()
    if progress == 1:
        print("")


__location__ = os.path.join(os.getcwd(), 'augmented_data', 'Stocks', '*.txt')
if not os.path.exists(os.path.join('result_images', 'polar')):
    os.makedirs(os.path.join('result_images', 'polar'))
if not os.path.exists(os.path.join('result_images', 'GAF')):
    os.makedirs(os.path.join('result_images', 'GAF'))

# Plot's specifics
font = {
    'family': 'serif',
    'color': 'darkblue',
    'weight': 'normal',
    'size': 16,
}

# to keep track of what's been worked on
total_files = len(os.listdir(os.path.dirname(__location__)))
i = 0

# for each file in folder
for fname in glob.glob(__location__):

    # Update the progress bar
    progress = float(i / total_files), (i + 1)
    updateProgress(progress[0], progress[1], total_files, os.path.basename(fname))

    # check that file is not empty
    if os.stat(fname).st_size != 0:
        # reading the source csv
        df = pd.read_csv(fname, header=0, parse_dates=[0], index_col=[0])

        # only want the close for this test
        df_close = df.drop(['Open', 'High', 'Low', 'Volume', 'OpenInt'], axis=1)

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
        plt.savefig(os.path.join(os.getcwd(), 'result_images', 'polar', os.path.basename(fname).replace('txt', 'png')),
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
        plt.savefig(os.path.join(os.getcwd(), 'result_images', 'GAF', os.path.basename(fname).replace('txt', 'png')),
                    bbox_inches='tight')

    i = i + 1

updateProgress(1, total_files, total_files, os.path.basename(fname))
