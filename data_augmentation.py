import glob
import os
from datetime import datetime
from progress_bar import updateProgress

import pandas as pd

# Parameters
# Rows to divide each csv file
N_ROWS = 365
# Date starting point
format_string = "%Y-%m-%d"
START_DATE_INDEX = datetime.strptime('2014-01-01', format_string)


def maxima_minima(row):
    if not pd.isna(row['min']):
        return 1
    if not pd.isna(row['max']):
        return -1
    else:
        return 0


# making sure writing directories exist
if not os.path.exists(os.path.join('augmented_data', 'Stocks')):
    os.makedirs(os.path.join('augmented_data', 'Stocks'))
if not os.path.exists(os.path.join('augmented_data', 'Labels')):
    os.makedirs(os.path.join('augmented_data', 'Labels'))

# setting read/write directory locations
__source_loc_ = os.path.join(os.getcwd(), 'Data', 'Stocks', '*.txt')
__write_root_loc__ = os.path.join(os.getcwd(), 'augmented_data', 'Stocks')

# to keep track of what's been worked on
total_files = len(os.listdir(os.path.dirname(__source_loc_)))

# for each file in folder
for i_originFile, fname in enumerate(glob.glob(__source_loc_)):
    if not os.path.exists(os.path.join('augmented_data', 'Stocks', os.path.basename(fname).replace(".txt", ""))):
        os.makedirs(os.path.join('augmented_data', 'Stocks', os.path.basename(fname).replace(".txt", "")))
    # Update the progress bar
    progress = float(i_originFile / total_files), (i_originFile + 1)
    updateProgress(progress[0], progress[1], total_files, os.path.basename(fname))

    # check that main file is not empty
    if os.stat(fname).st_size != 0:

        # reading the source csv
        original_df = pd.read_csv(fname, header=0, parse_dates=[0], index_col=[0])
        sliced_df = original_df.loc[START_DATE_INDEX:].copy()

        # calculating local minima and maxima
        sliced_df['min'] = sliced_df.Close[
            (sliced_df.Close.shift(1) > sliced_df.Close) & (sliced_df.Close.shift(-1) > sliced_df.Close)]
        sliced_df['max'] = sliced_df.Close[
            (sliced_df.Close.shift(1) < sliced_df.Close) & (sliced_df.Close.shift(-1) < sliced_df.Close)]
        # # Plot results
        # plt.scatter(df.loc['2000-01-01': '2001-01-01'].index, df.loc['2000-01-01': '2001-01-01']['min'], c='r')
        # plt.scatter(df.loc['2000-01-01': '2001-01-01'].index, df.loc['2000-01-01': '2001-01-01']['max'], c='g')
        # df.loc['2000-01-01': '2001-01-01'].Close.plot()
        # plt.show()

        # calculating the action column based on minima and maximas
        sliced_df['Action'] = sliced_df.apply(lambda row: maxima_minima(row), axis=1)

        # dropping unnecesary columns
        sliced_df = sliced_df.drop(['min', 'max'], axis=1)

        # generating subfiles from origin file
        fromRow = 0
        toRow = N_ROWS
        i_subFile = 1
        # old code, if unnecessary, will be deleted in a few commits
        # labels = list()
        # ids = list()
        files_index = pd.DataFrame(columns=['filename', 'action', 'close'])

        while toRow < sliced_df.shape[0]:
            # generating sub file
            sub_df = sliced_df.iloc[fromRow:toRow].copy()
            sub_df.to_csv(
                os.path.join(__write_root_loc__, os.path.basename(fname).replace(".txt", ""),
                             str(i_subFile) + "." + os.path.basename(fname)))

            # saving id and labels
            files_index.loc[fromRow] = [os.path.basename(
                os.path.join(__write_root_loc__, os.path.basename(fname).replace(".txt", ""),
                             str(i_subFile) + "." + os.path.basename(fname).replace(".txt", ".png"))),
                sliced_df.iloc[toRow - 1]['Action'],
                sliced_df.iloc[toRow - 1]['Close']]

            # old code, if unnecessary, will be deleted in a few commits
            # if fromRow != 0:
            #     labels.append(sliced_df.iloc[toRow - 1]['Action'])
            # if toRow + 1 < sliced_df.shape[0]:
            #     ids.append(
            #         os.path.basename(os.path.join(os.getcwd(),
            #                                       'augmented_data',
            #                                       'Stocks',
            #                                       str(i_subFile) + "_" + os.path.basename(fname)
            #                                       )
            #                          )
            #     )

            toRow = toRow + 1
            fromRow = fromRow + 1
            i_subFile = i_subFile + 1

        # writing labels set generated from i_originFile
        files_index.to_csv(
            os.path.join(os.getcwd(), 'augmented_data', 'Labels', os.path.basename(fname)), index=False)
        # old code, if unnecessary, will be deleted in a few commits
        # out = csv.writer(
        #     open(os.path.join(os.getcwd(), 'augmented_data', 'Labels', os.path.basename(fname)), "w"),
        #     delimiter='\n',
        #     quoting=csv.QUOTE_NONE)
        #
        # for idx, l_row in enumerate(labels):
        #     data = [str(ids[idx]).replace(".txt", ".png") + "," + str(l_row)]
        #     out.writerow(data)

    i_originFile = i_originFile + 1

updateProgress(1, total_files, total_files, os.path.basename(fname))
