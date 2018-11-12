import glob
import os
import sys

import pandas as pd


def maxima_minima(row):
    if not pd.isna(row['min']):
        return 1
    if not pd.isna(row['max']):
        return -1
    else:
        return 0


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


__location__ = os.path.join(os.getcwd(), 'Data', 'Stocks', '*.txt')
if not os.path.exists(os.path.join('augmented_data', 'Stocks')):
    os.makedirs(os.path.join('augmented_data', 'Stocks'))

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

        # calculating local minima and maxima
        df['min'] = df.Close[
            (df.Close.shift(1) > df.Close) & (df.Close.shift(-1) > df.Close)]
        df['max'] = df.Close[
            (df.Close.shift(1) < df.Close) & (df.Close.shift(-1) < df.Close)]
        # # Plot results
        # plt.scatter(df.loc['2000-01-01': '2001-01-01'].index, df.loc['2000-01-01': '2001-01-01']['min'], c='r')
        # plt.scatter(df.loc['2000-01-01': '2001-01-01'].index, df.loc['2000-01-01': '2001-01-01']['max'], c='g')
        # df.loc['2000-01-01': '2001-01-01'].Close.plot()
        # plt.show()

        # calculating the action column based on minima and maximas
        df['Action'] = df.apply(lambda row: maxima_minima(row), axis=1)

        # dropping unnecesary columns
        df = df.drop(['min', 'max'], axis=1)

        # saving results
        df.to_csv(os.path.join(os.getcwd(), 'augmented_data', 'Stocks', os.path.basename(fname)))
    i = i + 1

updateProgress(1, total_files, total_files, os.path.basename(fname))
