# Imports for P300
import multiprocessing
import time
from multiprocessing import Process, Value, Array, shared_memory, Manager
import multiprocessing.sharedctypes as sharedctypes
from multiprocessing.managers import BaseManager, NamespaceProxy

import ctypes
import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Style
import scipy.stats as stats
import subprocess, json

# Imports for OpenBCI
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, LogLevels
from brainflow.data_filter import DataFilter, FilterTypes, WindowFunctions, DetrendOperations
import csv


import logging
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from random import randint

import scipy.fft as fft

# # CODE FOR REAL TIME TEST # #

# # Create a Value data object # #
# This object can store a single integer and share it across multiple parallel processes
seconds = Value("i", 0)
counts = Value("i", 0)


# # Define Parallel Processes # #
         
# This function is the countup timer, the count is set to 0 before the script
# waits for a second, otherwise the beep will sound several times before the second 
# changes.
def timer(second, count, timestamps):
    # First we initialize a variable that will contain the moment the timer began and 
    # we store this in the timestamps list that will be stored in a CSV.    
    time_start = time.time()
    timestamps.append(time_start)
    while True:
        # The .get_lock() function is necessary since it ensures they are 
        # sincronized between both functions, since they both access to the same 
        # variables
        with second.get_lock(), count.get_lock():
            # We now calculate the time elappsed between start and now. 
            # (should be approx. 1 second)
            second.value = int(time.time() - time_start)
            count.value = 0
            if(second.value == 180):
                return
            print(second.value, end="\r")
        # Once we stored all the info and make the calculations, we sleep the script for
        # one second. This is the magic of the script, it executes every  ~1 second.
        time.sleep(1) #0.996

# This object is created to display in real time ENOPHONE's data, using self as argument, a variable that can be updated is created
# The argument board allow to call the specific board we are using in the main function to establish the connection.
class Graph:
    def __init__(self, board):
        #Data parameters to establish connections
        self.board_id = board.get_board_id()
        self.board_shim = board
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        #Calling the app inicialization to create a new window with plots
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="BOARD 1")

        # Functions inside the object to arrange data
        self._init_timeseries()
        self._init_processed()

        # Tools to start running the plot in real time
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.timeout.connect(self.preproccesing)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec_()


    #Plot for raw data
    def _init_timeseries(self):
        # Create an empty list to update data once the code start running
        self.plots = list()
        self.curves = list()
        for i in range(len(self.eeg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('Raw Data')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)
    #Plot for processed data
    def _init_processed(self):
        self.plots2 = list()
        self.curves2 = list()
        for i in range(len(self.eeg_channels)):
            p2 = self.win.addPlot(row=i, col=1)
            p2.showAxis('left', False)
            p2.setMenuEnabled('left', False)
            p2.showAxis('bottom', False)
            p2.setMenuEnabled('bottom', False)
            if i == 0:
                p2.setTitle('Processed Signal')
            self.plots2.append(p2)
            curve2 = p2.plot()
            self.curves2.append(curve2) 

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.eeg_channels):
            # plot timeseries
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()


    def preproccesing(self):
        data2 = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.eeg_channels):
            # plot timeseries
            DataFilter.detrend(data2[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data2[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data2[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data2[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves2[count].setData(data2[channel].tolist())

        self.app.processEvents()


class Graph2:
    def __init__(self, board2):
        self.board_id2 = board2.get_board_id()
        self.board_shim2 = board2
        self.eeg_channels2 = BoardShim.get_eeg_channels(self.board_id2)
        self.sampling_rate2 = BoardShim.get_sampling_rate(self.board_id2)
        self.update_speed_ms2 = 50
        self.window_size2 = 4
        self.num_points2 = self.window_size2 * self.sampling_rate2

        self.app2 = QtWidgets.QApplication([])
        self.win2 = pg.GraphicsLayoutWidget(show=True, title="BOARD 2")


        self._init_timeseries2()
        self._init_proccesed2()


        timer = QtCore.QTimer()
        timer.timeout.connect(self.update2)
        timer.timeout.connect(self.preproccesing2)
        timer.start(self.update_speed_ms2)
        QtWidgets.QApplication.instance().exec_()

    def _init_timeseries2(self):
        self.plotsa = list()
        self.curvesa = list()
        for i in range(len(self.eeg_channels2)):
            a = self.win2.addPlot(row=i, col=0)
            a.showAxis('left', False)
            a.setMenuEnabled('left', False)
            a.showAxis('bottom', False)
            a.setMenuEnabled('bottom', False)
            if i == 0:
                a.setTitle('Raw Data')
            self.plotsa.append(a)
            curvea = a.plot()
            self.curvesa.append(curvea)

    def _init_proccesed2(self):
        self.plotsa2 = list()
        self.curvesa2 = list()
        for i in range(len(self.eeg_channels2)):
            a2 = self.win2.addPlot(row=i, col=1)
            a2.showAxis('left', False)
            a2.setMenuEnabled('left', False)
            a2.showAxis('bottom', False)
            a2.setMenuEnabled('bottom', False)
            if i == 0:
                a2.setTitle('Proccesed Signal')
            self.plotsa2.append(a2)
            curvea2 = a2.plot()
            self.curvesa2.append(curvea2) 

    def update2(self):
        data3 = self.board_shim2.get_current_board_data(self.num_points2)
        for count, channel in enumerate(self.eeg_channels2):
            # plot timeseries
            self.curvesa[count].setData(data3[channel].tolist())

        self.app2.processEvents()


    def preproccesing2(self):
        data4 = self.board_shim2.get_current_board_data(self.num_points2)
        for count, channel in enumerate(self.eeg_channels2):
            # plot timeseries
            DataFilter.detrend(data4[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data4[channel], self.sampling_rate2, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data4[channel], self.sampling_rate2, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data4[channel], self.sampling_rate2, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curvesa2[count].setData(data4[channel].tolist())

        self.app2.processEvents()

######first try to put a graphic of bispectrum on real time###############
class Graph3:
    def __init__(self, df_gamma_average):
        #Data parameters to establish connections
        self.board_shim = df_gamma_average
        self.sampling_rate = 1
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        #Calling the app inicialization to create a new window with plots
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="BOARD 1")

        # Functions inside the object to arrange data
        self._init_timeseries()

        # Tools to start running the plot in real time
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec_()


    #Plot for raw data
    def _init_timeseries(self):
        # Create an empty list to update data once the code start running
        self.plots = list()
        self.curves = list()
        p = self.win.addPlot(row=0, col=0)
        p.showAxis('left', False)
        p.setMenuEnabled('left', False)
        p.showAxis('bottom', False)
        p.setMenuEnabled('bottom', False)
        self.plots.append(p)
        curve = p.plot()
        self.curves.append(curve)

    def update(self):
        data = self.board_shim
        self.curves.setData(data.tolist())

        self.app.processEvents()


##########################################################################

# # CODE FOR EEG # #
def EEG(second, folder, eno1_datach1, eno1_datach2):
    # The following object will save parameters to connect with the EEG.
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    # MAC Adress is the only required parameters for ENOPHONEs
    params.mac_address = 'f4:0e:11:75:75:a5'

    # Relevant board IDs available:
    board_id = BoardIds.ENOPHONE_BOARD.value # (37)
    #board_id = BoardIds.SYNTHETIC_BOARD.value # (-1)
    # board_id = BoardIds.CYTON_BOARD.value # (0)

    # Relevant variables are obtained from the current EEG.
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)

    # An empty dataframe is created to save Alpha/Beta values to plot in real time.
    #alpha_beta_data = pd.DataFrame(columns=['Alpha_C' + str(c) for c in range(1, len(eeg_channels) + 1)])
    ####################################################################

    ############# Session is then initialized #######################
    board.prepare_session()
    # board.start_stream () # use this for default options
    board.start_stream(45000, "file://{}/testOpenBCI.csv:w".format(folder))
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Starting the streaming with Enophones ---')

    try:
        while (True):
            time.sleep(4)
            data = board.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.

            ############## Data collection #################
            # Empty DataFrames are created for raw data.
            df_crudas = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            

            # The total number of EEG channels is looped to obtain MV for each channel, and
            # thus saved it on the corresponding columns of the respective DataFrame.
            for eeg_channel in eeg_channels:
                DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
                ####################START OF PREPROCESING#############################
                #Filter for envirionmental noise (Notch: 0=50Hz 1=60Hz)
                DataFilter.remove_environmental_noise(data[eeg_channel], sampling_rate, noise_type=1)
                #Bandpass Filter
                DataFilter.perform_lowpass(data[eeg_channel], sampling_rate, cutoff=100, order=4, filter_type=0, ripple=0)
                DataFilter.perform_highpass(data[eeg_channel], sampling_rate, cutoff=0.1, order=4, filter_type=0, ripple=0)      
                df_crudas['MV' + str(eeg_channel)] = data[eeg_channel]
                #df_bispec['BISPEC' + str(eeg_channel)] = data[eeg_channel]
            # Calculate the new variable based on the formula
            referenced_electrodes = pd.DataFrame()
            referenced_electrodes['referenced_electrode1'] = df_crudas['MV3'] - ((df_crudas['MV1'] + df_crudas['MV2']) / 2)
            referenced_electrodes['referenced_electrode2'] = df_crudas['MV4'] - ((df_crudas['MV1'] + df_crudas['MV2']) / 2)

            # Both raw and PSD DataFrame is exported as a CSV.
            arrange=referenced_electrodes.to_dict('dict')

            
            info1=arrange['referenced_electrode1']
            info2=arrange['referenced_electrode2']
            #info3=arrange['MV3']
            #info4=arrange['MV4']

            lista1 = list(info1.values())
            lista2 = list(info2.values())
            #lista3 = list(info3.values())
            #lista4 = list(info4.values())

            eno1_datach1[:800] = lista1[:800]
            eno1_datach2[:800] = lista2[:800]
            #eno1_datach3[:800] = lista3[:800]
            #eno1_datach4[:800] = lista4[:800]


            df_crudas.to_csv('{}/Raw/Crudas.csv'.format(folder), mode='a')

            #Uncomment the line below if you want to se the real time graphics of the preprocessing stage, it may cause problems in code efficiency and shared memory
            #Graph(board)
            with second.get_lock():
                # When seconds reach the value, we exit the functions.
                if(second.value == 180):
                    plt.close()
   
                    return
            

    except KeyboardInterrupt:
        board.stop_stream()
        board.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- End the session with Enophones ---')

    ##############Links que pueden ayudar al entendimiento del código ##############
    # https://www.geeksforgeeks.org/python-save-list-to-csv/

# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.

######################################################## ENOPHONES 2 ###############################################################################
def EEG2(second, folder, eno2_datach1, eno2_datach2):
    # PSD 128 rows
    # Band power 1 row each

    # The following object will save parameters to connect with the EEG.
    BoardShim.enable_dev_board_logger()
    params2 = BrainFlowInputParams()

    # MAC Adress is the only required parameters for ENOPHONEs
    params2.mac_address = 'f4:0e:11:75:75:ce'

    # Relevant board IDs available:
    board_id2 = BoardIds.ENOPHONE_BOARD.value # (37)
    #board_id2 = BoardIds.SYNTHETIC_BOARD.value # (-1)
    # board_id = BoardIds.CYTON_BOARD.value # (0)

    # Relevant variables are obtained from the current EEG.
    eeg_channels2 = BoardShim.get_eeg_channels(board_id2)
    sampling_rate2 = BoardShim.get_sampling_rate(board_id2)
    board2 = BoardShim(board_id2, params2)


    ############# Session is then initialized #######################
    board2.prepare_session()
    # board.start_stream () # use this for default options
    board2.start_stream(45000, "file://{}/testOpenBCI2.csv:w".format(folder))
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Starting the streaming with Enophones 2 ---')

    try:
        
        while (True):
            time.sleep(4)
            with second.get_lock():
                # When the seconds reach 312, we exit the functions.
                if(second.value == 180):
                    plt.close()
                    return
            data2 = board2.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.

            ############## Data collection #################


            # Empty DataFrames are created for raw data.
            df_crudas2 = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels2) + 1)])

            # The total number of EEG channels is looped to obtain MV and PSD for each channel, and
            # thus saved it on the corresponding columns of the respective DataFrame.
            for eeg_channel2 in eeg_channels2:
                DataFilter.detrend(data2[eeg_channel2], DetrendOperations.LINEAR.value)
                #Filter for envirionmental noise (Notch: 0=50Hz 1=60Hz)
                DataFilter.remove_environmental_noise(data2[eeg_channel2], sampling_rate2, noise_type=1)
                #Bandpass Filter
                DataFilter.perform_lowpass(data2[eeg_channel2], sampling_rate2, cutoff=100, order=4, filter_type=0, ripple=0)
                DataFilter.perform_highpass(data2[eeg_channel2], sampling_rate2, cutoff=0.1, order=4, filter_type=0, ripple=0)
                df_crudas2['MV' + str(eeg_channel2)] = data2[eeg_channel2]

            # Calculate the new variable based on the formula
            referenced_electrodes2 = pd.DataFrame()
            referenced_electrodes2['referenced_electrode1'] = df_crudas2['MV3'] - ((df_crudas2['MV1'] + df_crudas2['MV2']) / 2)
            referenced_electrodes2['referenced_electrode2'] = df_crudas2['MV4'] - ((df_crudas2['MV1'] + df_crudas2['MV2']) / 2)

 

            arrange2=referenced_electrodes2.to_dict('dict')

            
            info12=arrange2['referenced_electrode1']
            info22=arrange2['referenced_electrode2']
            #info32=arrange2['MV3']
            #info42=arrange2['MV4']

            lista12 = list(info12.values())
            lista22 = list(info22.values())
            #lista32 = list(info32.values())
            #lista42 = list(info42.values())

            
            eno2_datach1[:800] = lista12[:800]
            eno2_datach2[:800] = lista22[:800]
            #eno2_datach3[:800] = lista32[:800]
            #eno2_datach4[:800] = lista42[:800]



            # Both the raw and PSD DataFrame is exported as a CSV.
            df_crudas2.to_csv('{}/Raw 2/Crudas2.csv'.format(folder), mode='a')

            #Uncomment the line below if you want to se the real time graphics of the preprocessing stage, it may cause problems in code efficiency and shared memory
            #Graph2(board2)

    except KeyboardInterrupt:
        board2.stop_stream()
        board2.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- End the session with Enophone 2 ---')


    ##############Links que pueden ayudar al entendimiento del código ##############
    # https://www.geeksforgeeks.org/python-save-list-to-csv/

# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.
def bispec(eno1_datach1, eno1_datach2, eno2_datach1, eno2_datach2, second, folder):
    try:
        while (True):
            time.sleep(4)
            df_bispecMV1=eno1_datach1
            df_bispecMV2=eno1_datach2


            df_bispec2MV1=eno2_datach1
            df_bispec2MV2=eno2_datach2


            matrix_eno1=np.array([df_bispecMV1[:], df_bispecMV2[:]])
            matrix_eno1t=matrix_eno1.transpose()
            matrix_eno2=np.array([df_bispec2MV1[:], df_bispec2MV2[:]])
            matrix_eno2t=matrix_eno2.transpose()


            cont = 0
            Nch=2

            B = np.zeros((Nch*Nch, len(df_bispecMV1)//2))
            index = np.zeros((Nch*Nch, 2))
            

            for ch2 in range(Nch):
                for ch1 in range(Nch):
                    bs = np.abs(np.fft.fft(matrix_eno1t[:, ch1])*np.fft.fft(matrix_eno2t[:, ch2])*np.conj(np.fft.fft(matrix_eno1t[:, ch1]+matrix_eno2t[:, ch2])))
                    B[cont, :] = np.log(bs[:len(bs)//2].T)  # Mean windows bs on all channels
                    index[cont, :] = [ch1+1, ch2+1]  # Indexing combination order: ch1,ch2
                    cont += 1
                
                #df_time[Nch] = B[Nch]
            print(B)
            
            
            bispectrum = pd.DataFrame(B)
            b_transpose = bispectrum.transpose()


            df_bispec = pd.DataFrame(columns=['COMB' + str(channel) for channel in range(0, len(index))])
            for eeg_channel2 in range (0,4):
                df_bispec['COMB' + str(eeg_channel2)] = b_transpose[eeg_channel2]
            df_norm = np.zeros((len(df_bispec), Nch*Nch))
            #print(df_bispec)
            #df_norm = pd.DataFrame()
            df_bispec.to_csv('{}/Bispec.csv'.format(folder), mode='a')
            #df_norm = pd.DataFrame()

            

            with second.get_lock():
                # When the seconds reach 312, we exit the functions.
                if(second.value == 180):
                    plt.close()
                    return
                elif ((second.value > 4) and (second.value < 60)):
                    #Get data to apply normalization
                    for i in range (1):
                        print('Preparing device calibration...')
                        df_eo = df_bispec
                        df_eo.to_csv('{}/Calibration_data.csv'.format(folder), mode='a')

                elif ((second.value > 63) and (second.value<180)):
                        #Create dataframes to estimate the eyes open mean matrix

                        sum = pd.read_csv('{}/Calibration_data.csv'.format(folder), index_col=0)
                        #eyes_open = np.zeros((800, 16))
                        #for i in sum:
                            #arrange3=pd.to_numeric(sum[i], errors='coerce')#.dropna(axis=0).reset_index(drop=True)
                        arrange3 = sum.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True)

                            #matrix = pd.DataFrame(arrange3).transpose()
                        arrange3.to_csv('{}/Calibration_data_clean.csv'.format(folder))

                          
                        eyes_open = pd.read_csv('{}/Calibration_data_clean.csv'.format(folder), index_col=0)
                        
                        df_eo = pd.DataFrame(eyes_open)
                        divisor = len(df_eo)/len(df_bispec)
                        df_eo2 = df_eo.rename(columns={'COMB0': 0, 'COMB1': 1, 'COMB2': 2, 'COMB3': 3, 'COMB4': 4, 'COMB5': 5, 'COMB6': 6, 'COMB7': '7', 'COMB8': 8, 'COMB9': 9, 'COMB10': 10, 'COMB11': 11, 'COMB12': 12, 'COMB13': 13, 'COMB14': 14, 'COMB15': 15})
                        dic_eo = df_eo2.to_dict('dict')
                    
    

                        # Create an array to store the relevant keys
                        relevant_keys = np.arange(0, len(df_eo), 400)
                        
                        for i in range(400):
                            for comb, bis in dic_eo.items():
                                # Calculate the indices to access values in bis
                                indices = relevant_keys + i
                                # Sum the relevant values using NumPy's array operations
                                sum_values = np.sum([bis[key] for key in indices])
                                df_norm[i, int(comb)] = sum_values / divisor
                                    
            df_sum = pd.DataFrame(df_norm)

            df_sum2 = df_sum.rename(columns={0: 'COMB0', 1: 'COMB1', 2: 'COMB2', 3: 'COMB3', 4: 'COMB4', 5: 'COMB5', 6: 'COMB6', 7: 'COMB7', 8: 'COMB8', 9: 'COMB9', 10: 'COMB10', 11: 'COMB11', 12: 'COMB12', 13: 'COMB13', 14: 'COMB14', 15: 'COMB15'})
            df_sub = df_bispec.sub(df_sum2)
            df_div = df_sub.div(df_sum2)
            print(df_div)



            #Get frequency bands to apply in bispectrum matrix normalized
            delta_limit = (4 * len(df_bispec)) // 125 #125Hz is the frequency limit to the bispectrum matrix length data
            theta_limit = (8 * len(df_bispec)) // 125
            alpha_limit = (13 * len(df_bispec)) // 125
            beta_limit = (29 * len(df_bispec)) // 125
            gamma_limit = (50 * len(df_bispec)) // 125
            

            df_delta = df_div.iloc[0:delta_limit, :].mean(axis=0)
            df_theta = df_div.iloc[delta_limit:theta_limit, :].mean(axis=0)
            df_alpha = df_div.iloc[theta_limit:alpha_limit, :].mean(axis=0)
            df_beta = df_div.iloc[alpha_limit:beta_limit, :].mean(axis=0)
            df_gamma = df_div.iloc[beta_limit:gamma_limit, :].mean(axis=0)
            print(df_gamma)

            # Concatenate the individual DataFrames horizontally (column-wise)
            result_df = pd.concat([df_delta, df_theta, df_alpha, df_beta, df_gamma], axis=0)

            # Transpose the concatenated DataFrame to have a shape of [1 row x 20 columns]
            bispectrum_mean = pd.DataFrame(result_df).transpose()

            # Create a list of new column names with both the combination number and frequency band
            new_column_names = []

            # Define the frequency bands
            frequency_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

            # Loop through the combination numbers and frequency bands to create new column names
            for band in frequency_bands:
                for comb_num in range(1,5):
                    new_column_names.append(f'COMB{comb_num}_{band}')

            # Assign the new column names to the DataFrame
            bispectrum_mean.columns = new_column_names

                            #matrix = pd.DataFrame(arrange3).transpose()
            bispectrum_mean.to_csv('{}/Frequency_bands_bispectrum.csv'.format(folder), mode='a')



            print(bispectrum_mean)
            #Graph3(df_gamma_average)

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- End the session with Enophone 2 ---')

###################################################################################################################################################
if __name__ == '__main__':
    # Access to Manager to share memoru between proccesses and acces dataframe's 
    mgr = Manager()
    #ns = mgr.list()
    eno1_datach1 = multiprocessing.Array('d', 800)
    eno1_datach2 = multiprocessing.Array('d', 800)


    eno2_datach1 = multiprocessing.Array('d', 800)
    eno2_datach2 = multiprocessing.Array('d', 800)



    # # Define the data folder # #
    # The name of the folder is defined depending on the user's input
    subject_ID, repetition_num = input('Please enter the subject ID and the number of repetition: ').split(' ')
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    repetition_num = '0' + repetition_num if int(repetition_num) < 10 else repetition_num
    folder = 'S{}R{}_{}'.format(subject_ID, repetition_num, datetime.now().strftime("%d%m%Y_%H%M"))
    os.mkdir(folder)


    for subfolder in ['Raw', 'Processed', 'Figures']:
        os.mkdir('{}/{}'.format(folder, subfolder))

    #Creación de carpetas para datos de Enophones 2
    for subfolder2 in ['Raw 2', 'Processed 2', 'Figures 2']:
        os.mkdir('{}/{}'.format(folder, subfolder2))

    # # Create a multiprocessing List # # 
    # This list will store the seconds where a beep was played
    timestamps = multiprocessing.Manager().list()
    
    # # Start processes # #

    process2 = Process(target=timer, args=[seconds, counts, timestamps])
    q = Process(target=EEG, args=[seconds, folder, eno1_datach1, eno1_datach2])
    q2 = Process(target=EEG2, args=[seconds, folder, eno2_datach1, eno2_datach2])
    q3 = Process(target=bispec, args=[eno1_datach1, eno1_datach2, eno2_datach1, eno2_datach2, seconds, folder])


    process2.start()
    q.start()
    q2.start()
    q3.start()


    process2.join()
    q.join()
    q2.join()
    q3.join()


    # # DATA STORAGE SECTION # #
    # Executed only once the test has finished.
    
    print(Fore.RED + 'Test finished sucessfully, storing data now...' + Style.RESET_ALL)


    print(Fore.GREEN + 'Data stored sucessfully' + Style.RESET_ALL)
    
    # # Data processing # #
    print(Fore.RED + 'Data being processed...' + Style.RESET_ALL)


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
    # It reads the files saved in the "Raw" folder, and only reads .CSV files, to outputt a .CSV file in "Processed" folder.
    for df_name in os.listdir('{}/Raw/'.format(folder)):
        if df_name[-4:] == '.csv' and df_name[:4] != 'file':
            df_name = df_name[:-4]
            df_raw = pd.read_csv('{}/Raw/{}.csv'.format(folder, df_name), index_col=0)
            df_processed = remove_outliers(df_raw.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True), 'quantile')
            
            # The processed DataFrame is then exported to the "Processed" folder, and plotted.
            df_processed.to_csv('{}/Processed/{}_processed.csv'.format(folder, df_name))
            df_processed.plot()

            # The plot of the processed DataFrame is saved in the "Figures" folder.
            plt.savefig('{}/Figures/{}_plot.png'.format(folder, df_name))

    for df_name2 in os.listdir('{}/Raw 2/'.format(folder)):
        if df_name2[-4:] == '.csv' and df_name2[:4] != 'file':
            df_name2 = df_name2[:-4]
            df_raw2 = pd.read_csv('{}/Raw 2/{}.csv'.format(folder, df_name2), index_col=0)
            df_processed2 = remove_outliers(df_raw2.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True), 'quantile')
            
            # The processed DataFrame is then exported to the "Processed" folder, and plotted.
            df_processed2.to_csv('{}/Processed 2/{}_processed2.csv'.format(folder, df_name2))
            df_processed2.plot()

            # The plot of the processed DataFrame is saved in the "Figures" folder.
            plt.savefig('{}/Figures 2/{}_plot2.png'.format(folder, df_name2))

    print(Fore.GREEN + 'Data processed successfully' + Style.RESET_ALL)

                        #Create dataframes to estimate the eyes open mean matrix

    data_meanb = pd.read_csv('{}/Frequency_bands_bispectrum.csv'.format(folder), index_col=0)
    data_graph = data_meanb.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True)

                            #matrix = pd.DataFrame(arrange3).transpose()
    #arrange3.to_csv('{}/Calibration_data_clean.csv'.format(folder))

                          
    #eyes_open = pd.read_csv('{}/Calibration_data_clean.csv'.format(folder), index_col=0)
                        
    df_graph = pd.DataFrame(data_graph)
    # Replace -inf and inf with 0 in your DataFrame
    data_graph = data_graph.replace([float('-inf'), float('inf')], 0)
    print(df_graph)

    # Assuming data_graph is your DataFrame
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


####### Sources ########
# To understand Value data type and lock method read the following link:
# https://www.kite.com/python/docs/multiprocessing.Value  
# For suffle of array, check the next link and user "mdml" answer:
# https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones



#python Empatica-Project-ALAS-main/files/final_bispec.py