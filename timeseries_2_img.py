import os
import math
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import numpy as np
import glob
import gramian_angular_field
import tools


# normalizing values
def rescale(x):
    return ((x - max_close) + (x - min_close)) / (max_close - min_close)


def arccosine(x):
    return np.arccos(x)


def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))


def cos_sum(a, b):
    """To work with tabulate."""
    return (math.cos(a + b))


__location__ = os.path.join(os.getcwd(), 'Data', 'Stocks', '*.txt')
if not os.path.exists('images'):
    os.makedirs('images')

# Instanciate figure
fig = plt.figure(figsize=(9, 6))

size = 0.33
alignement = 0.1

# Classic plot
ax_carthesian = fig.add_axes([alignement, 0.4, size, size])
# Polar plot
ax_polar = fig.add_axes([alignement + size, 0.4, size, size], polar=True)
# Patchwork
ax_patchwork = fig.add_axes([alignement + 1.8 * size, 0.4, size, size])

# Global iteration
iteration = 0

# PLOTS
global size_time_serie
size_time_serie = float(45)

# for each file in folder
for fname in glob.glob(__location__):
    # check that file is not empty
    if os.stat(fname).st_size != 0:
        # reading the source csv
        df = pd.read_csv(fname, header=0, parse_dates=[0], index_col=[0])
        # only want the close for this test
        df_close = df.drop(['Open', 'High', 'Low', 'Volume', 'OpenInt'], axis=1)

        # applying normalization function
        min_close = df_close.min()
        max_close = df_close.max()
        df_close_normalized = df_close['Close'].apply(rescale)
        df_close_normalized_arccosine = df_close_normalized.apply(arccosine)
        t = mdates.date2num(df_close_normalized.index.to_pydatetime())
        y = df_close_normalized_arccosine['Close']
        # Get data
        gaf, phi, r, scaled_time_serie = gramian_angular_field(y)
        print(gaf, phi, r, scaled_time_serie)
        # ax = plt.subplot(111, projection='polar')
        # tnorm = (t - t.min()) / (t.max() - t.min()) * 2. * np.pi
        # ax.plot(tnorm, y, linewidth=0.8)
        # # ax.set_rmax(2)
        # # ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
        # # ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
        # # ax.grid(True)
        # ax.set_title("A line plot on a polar axis", va='bottom')
        # plt.savefig(os.path.join(os.getcwd(), 'images', os.path.basename(fname).replace('txt', 'png')))
        # plt.gcf().clear()
