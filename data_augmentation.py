import glob
import os
import sys

import pandas as pd
import csv

# Rows to divide each csv file
N_ROWS = 7

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
    line = str("\rStock file: {0}/{1} [{2}] {3}% {4}").format(
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
if not os.path.exists(os.path.join('augmented_data', 'Labels')):
    os.makedirs(os.path.join('augmented_data', 'Labels'))

# to keep track of what's been worked on
total_files = len(os.listdir(os.path.dirname(__location__)))


# for each file in folder
for i_originFile, fname in enumerate(glob.glob(__location__)):

    # Update the progress bar
    progress = float(i_originFile / total_files), (i_originFile + 1)
    updateProgress(progress[0], progress[1], total_files, os.path.basename(fname))

    # check that main file is not empty
    if os.stat(fname).st_size != 0:

        # reading the source csv
        #dfTotal = pd.read_csv(fname, header=0, parse_dates=[0], index_col=[0])
        origin_df = pd.read_csv(fname, header=0, parse_dates=[0], index_col=[0])

        # calculating local minima and maxima
        origin_df['min'] = origin_df.Close[
            (origin_df.Close.shift(1) > origin_df.Close) & (origin_df.Close.shift(-1) > origin_df.Close)]
        origin_df['max'] = origin_df.Close[
            (origin_df.Close.shift(1) < origin_df.Close) & (origin_df.Close.shift(-1) < origin_df.Close)]
        # # Plot results
        # plt.scatter(df.loc['2000-01-01': '2001-01-01'].index, df.loc['2000-01-01': '2001-01-01']['min'], c='r')
        # plt.scatter(df.loc['2000-01-01': '2001-01-01'].index, df.loc['2000-01-01': '2001-01-01']['max'], c='g')
        # df.loc['2000-01-01': '2001-01-01'].Close.plot()
        # plt.show()

        # calculating the action column based on minima and maximas
        origin_df['Action'] = origin_df.apply(lambda row: maxima_minima(row), axis=1)

        # dropping unnecesary columns
        origin_df = origin_df.drop(['min', 'max'], axis=1)

        # generating subfiles from origin file
        fromRow = 0
        toRow = N_ROWS
        i_subFile = 1
        labels = list()
        ids = list()
        while toRow < origin_df.shape[0]:
            # generating sub file
            sub_df = origin_df.iloc[fromRow:toRow].copy()
            sub_df.to_csv(os.path.join(os.getcwd(), 'augmented_data', 'Stocks', str(i_subFile)+"_"+os.path.basename(fname)))

            # saving id and label
            if fromRow != 0:
                labels.append(origin_df.iloc[toRow-1]['Action'])
            if toRow+1 < origin_df.shape[0]:
                ids.append(
                    os.path.basename(os.path.join(os.getcwd(),
                                                  'augmented_data',
                                                  'Stocks',
                                                  str(i_subFile) + "_" + os.path.basename(fname)
                                                  )
                                     )
                )
            toRow = toRow + 1
            fromRow = fromRow + 1
            i_subFile = i_subFile + 1

        # writing labels set generated from i_originFile
        out = csv.writer(
            open(os.path.join(os.getcwd(), 'augmented_data', 'Labels', os.path.basename(fname)), "w"),
            delimiter='\n',
            quoting=csv.QUOTE_NONE)

        for idx, l_row in enumerate(labels):
            data = [str(ids[idx]).replace(".txt", ".png") + ", " + str(l_row)]
            out.writerow(data)

    i_originFile = i_originFile + 1


updateProgress(1, total_files, total_files, os.path.basename(fname))
