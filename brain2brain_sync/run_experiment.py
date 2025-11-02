# Imports from brain2brain_sync module
from EEG_device import EEG
from bispectrum import bispec
from stopwatch import timer


# Imports for multiprocessing and board shim connection
from multiprocessing import Process, Value, Manager, Array
from brainflow.board_shim import BoardIds

# Imports for folders creation and data storage
import os
from datetime import datetime
from colorama import Fore, Style

# Imports for outliers removal and data plotting (no real-time)
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


# # CODE FOR REAL TIME TEST # #

# # Create a Value data object # #
# This object can store a single integer and share it across multiple parallel processes
seconds = Value("i", 0)
counts = Value("i", 0)


# Choose the board ID here, once
# board_id = BoardIds.ENOPHONE_BOARD.value
board_id = BoardIds.SYNTHETIC_BOARD.value


###################################################################################################################################################
if __name__ == '__main__':
    # Access to Manager to share memory between proccesses and acces dataframe's 
    mgr = Manager()

    eno1_datach1 = Array('d', 800)
    eno1_datach2 = Array('d', 800)
    eno2_datach1 = Array('d', 800)
    eno2_datach2 = Array('d', 800)

    # Write specfic MAC addresses for each device
    mac1 = "f4:0e:11:75:75:a5"
    mac2 = "aa:bb:cc:dd:ee:ff"  

    # # Define the data folder # #
    # The name of the folder is defined depending on the user's input

    while True:
        try:
            dyad = int(input("Please write the assigned number for the dyad under analysis: "))
            break  # exit loop if input is valid
        except ValueError:
            print("Invalid input. Please enter a whole number (integer).")

    while True:
        try:
            repetition_num = int(input("Enter the iteration number of the current experimental test: "))
            break
        except ValueError:
            print("Invalid input. Please enter a whole number (integer).")

    dyad = f"{dyad:02d}"
    repetition_num = f"{repetition_num:02d}"
    folder = f"experimental_results/Dyad{dyad}/R{repetition_num}_{datetime.now():%d%m%Y_%H%M}"
    os.makedirs(folder, exist_ok=True)


    for subfolder in ['Real_Time_Data', 'Processed', 'Figures']:
        os.mkdir('{}/{}'.format(folder, subfolder))

    # # Create a multiprocessing List # # 
    timestamps = Manager().list()

    # # Start processes # #
    counter = Process(target=timer, args=[seconds, counts, timestamps])
    subject1 = Process(target=EEG, args=[seconds, folder, eno1_datach1, eno1_datach2, mac1, "Device_1", board_id])
    subject2 = Process(target=EEG, args=[seconds, folder, eno2_datach1, eno2_datach2, mac2, "Device_2", board_id])
    bispectrum = Process(target=bispec, args=[eno1_datach1, eno1_datach2, eno2_datach1, eno2_datach2, seconds, folder])

    counter.start()
    subject1.start()
    subject2.start()
    bispectrum.start()

    counter.join()
    subject1.join()
    subject2.join()
    bispectrum.join()


    # # DATA STORAGE SECTION # #
    # Executed only once the test has finished.
    print(Fore.RED + 'Test finished successfully, storing data now...' + Style.RESET_ALL)
    print(Fore.GREEN + 'Data stored successfully' + Style.RESET_ALL)

    # # Data processing # #
    print(Fore.RED + 'Data being processed...' + Style.RESET_ALL)


    # # POST REAL-TIME OUTLIERS REMOVAL SECTION # #
    def remove_outliers(df, method):
        """
        Uses an statistical method to remove outlier rows from the dataset x, and filters the valid rows back to y.

        :param pd.DataFrame df: with non-normalized, source variables.
        :param string method: type of statistical method used.
        :return pd.DataFrame: Filtered DataFrame
        """

        # The number of initial rows is saved.
        n_pre = df.shape[0]

        # A switch case selects an statistical method to remove rows considered as outliers.
        if method == 'z-score':
            z = np.abs(stats.zscore(df))
            df = df[(z < 3).all(axis=1)]
        elif method == 'quantile':
            q1 = df.quantile(q=.25)
            q3 = df.quantile(q=.75)
            iqr = df.apply(stats.iqr)
            df = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
        
        # The difference between the processed and raw rows is printed.
        n_pos = df.shape[0]
        diff = n_pre - n_pos
        print(f'{diff} rows removed {round(diff / n_pre * 100, 2)}%')
        return df
    
    # The following for loop iterates over all features, and removes outliers depending on the statistical method used.
    for df_name in os.listdir('{}/Real_Time_Data/'.format(folder)):
        if df_name[-4:] == '.csv' and df_name[:4] != 'file':
            df_name = df_name[:-4]
            df_raw = pd.read_csv('{}/Real_Time_Data/{}.csv'.format(folder, df_name), index_col=0)
            df_processed = remove_outliers(df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True), 'quantile')
            
            # The processed DataFrame is then exported to the "Processed" folder, and plotted.
            df_processed.to_csv('{}/Processed/{}_processed.csv'.format(folder, df_name))
            df_processed.plot()

            # The plot of the processed DataFrame is saved in the "Figures" folder.
            plt.savefig('{}/Figures/{}_plot.png'.format(folder, df_name))


    print(Fore.GREEN + 'Data processed successfully' + Style.RESET_ALL)


    # # POST REAL-TIME BISPECTRUM PLOTS SECTION # #
    #Create dataframes for bispectrum results
    data_meanb = pd.read_csv('{}/Frequency_bands_bispectrum.csv'.format(folder), index_col=0)
    data_graph = data_meanb.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True)
                        
    df_graph = pd.DataFrame(data_graph)
    # Replace -inf and inf with 0 in your DataFrame
    data_graph = data_graph.replace([float('-inf'), float('inf')], 0)
    print(df_graph)
'''
    # Generate a time index from 0 to 420 seconds with the same length as your DataFrame
    time_index = np.linspace(0, 180, len(data_graph))
    print(seconds)
    print(timestamps)
    for column in data_graph.columns:
        plt.figure(figsize=(8, 6))
        plt.plot(time_index, data_graph[column], label=column)
        plt.title(f'Plot of {column}')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{folder}/Figures/{column}_plot.png')
        plt.show()
'''

####### Sources ########
# To understand Value data type and lock method read the following link:
# https://www.kite.com/python/docs/multiprocessing.Value  
# For suffle of array, check the next link and user "mdml" answer:
# https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones

