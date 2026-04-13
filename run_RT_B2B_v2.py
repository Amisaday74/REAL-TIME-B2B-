# Imports from brain2brain_sync module
from brain2brain_sync import EEG, bispec, timer, Graph

# Imports for multiprocessing and board shim connection
from multiprocessing import Process, Value, Manager, Array, Queue, Event, Lock
from brainflow.board_shim import BoardIds, BoardShim

# Imports for folders creation and data storage
import os
import json
from datetime import datetime
from colorama import Fore, Style

# Imports for outliers removal and data plotting (no real-time)
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, QtCore
import sys
import threading


# # CODE FOR REAL TIME TEST # #

# # Load configuration from JSON file # #
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# # Create a Value data object # #
# This object can store a single integer and share it across multiple parallel processes
seconds = Value("i", 0)


# Choose the board ID from config
board_id_name = config['board_id']
board_id = getattr(BoardIds, board_id_name).value
eeg_channels = BoardShim.get_eeg_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)
test_duration = config['test_duration_seconds']
timewindow = config['timewindow_seconds']
bispectrum_channels = config['channels_for_bispectrum']
experiment_phase = config['experiment_phase']

def poll_queues(graph1, graph2, queues, device_1_name, device_2_name):
    for q in queues:
        while not q.empty():
            device, raw, processed = q.get()
            if device == device_1_name:
                if graph1.running:
                    graph1.data_signal.emit(raw)
                    graph1.processed_data.emit(processed)
            elif device == device_2_name:
                if graph2.running:
                    graph2.data_signal.emit(raw)
                    graph2.processed_data.emit(processed)


###################################################################################################################################################
if __name__ == '__main__':
    # Access to Manager to share memory between proccesses and acces dataframe's 
    mgr = Manager()

    BUFFER_LEN = sampling_rate * timewindow * 3  # Buffer to store 3 time windows of data (e.g., 3000 data points for 12 seconds if timewindow is 4s and sampling_rate is 1000Hz)
    N_CH = bispectrum_channels  # Number of channels to store (referenced electrodes)
    bispectrum_length = sampling_rate * timewindow // 2  # Length of bispectrum output (half the window size due to FFT symmetry)

    # -------------------------
    # Shared ring buffers
    # -------------------------
    eno1_buffer_raw = Array('d', N_CH * BUFFER_LEN, lock=False)
    eno2_buffer_raw = Array('d', N_CH * BUFFER_LEN, lock=False)

    # Write indices
    eno1_write_idx = Value('i', 0)
    eno2_write_idx = Value('i', 0)

    # Locks (single-writer, multi-reader safe)
    eno1_lock = Lock()
    eno2_lock = Lock()



    # Load MAC addresses from configuration
    mac1 = config['devices'][0]['mac_address']
    mac2 = config['devices'][1]['mac_address']
    device_1_name = config['devices'][0]['device_name']
    device_2_name = config['devices'][1]['device_name']  

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
    if experiment_phase == "calibration":
        folder = f"experimental_results/Dyad{dyad}/Calibration_data"
    elif experiment_phase == "interaction":
        folder = f"experimental_results/Dyad{dyad}/R{repetition_num}_{datetime.now():%d%m%Y_%H%M}"
    os.makedirs(folder, exist_ok=True)


    for subfolder in ['Real_Time_Data', 'Bispectrum', 'Figures']:
        os.mkdir('{}/{}'.format(folder, subfolder))

    # # Create a multiprocessing List # # 
    timestamps = Manager().list()

    # Create events for synchronization
    event1 = Event()
    event2 = Event()
    completion_event = Event()  # Event to signal when stopwatch reaches 20s

    q1 = Queue()
    q2 = Queue()
    graph_closed = threading.Event()

    # # Start processes # #
    counter = Process(target=timer, args=[seconds, completion_event, test_duration])
    subject1 = Process(target=EEG, args=[seconds, folder, eno1_buffer_raw, eno1_write_idx, eno1_lock, mac1, device_1_name, board_id, q1, event1, completion_event, timewindow])
    subject2 = Process(target=EEG, args=[seconds, folder, eno2_buffer_raw, eno2_write_idx, eno2_lock, mac2, device_2_name, board_id, q2, event2, completion_event, timewindow])
    bispectrum = Process(target=bispec, args=[eno1_buffer_raw, eno1_write_idx, eno1_lock, eno2_buffer_raw, eno2_write_idx, eno2_lock, seconds, folder, event1, event2, completion_event, bispectrum_channels, experiment_phase, sampling_rate, timewindow])

    counter.start()
    subject1.start()
    subject2.start()
    bispectrum.start()

    # -----------------------------------------------------------
    # Qt GUI runs in main thread (no warning, responsive)
    # -----------------------------------------------------------
    app = QtWidgets.QApplication(sys.argv)
    graph1 = Graph(eeg_channels, sampling_rate, "Device 1 EEG Data")
    graph2 = Graph(eeg_channels, sampling_rate, "Device 2 EEG Data")

    # Poll incoming data from queues every 100 ms
    q_timer = QtCore.QTimer()
    q_timer.timeout.connect(lambda: poll_queues(graph1, graph2, [q1, q2], device_1_name, device_2_name))
    q_timer.start(100)

    # Monitor subprocesses; close GUI when all are done
    def check_workers():
        alive = any(p.is_alive() for p in [counter, subject1, subject2, bispectrum])
        if not alive:
            print("All processes finished — closing GUI.")
            graph1.close_app()
            graph2.close_app()          # emits quit on QApplication
    monitor_timer = QtCore.QTimer()
    monitor_timer.timeout.connect(check_workers)
    monitor_timer.start(500)

    # Start Qt event loop (blocking but responsive)
    app.exec_()
    print("Graph window closed — continuing post-processing.")

    counter.join()
    subject1.join()
    subject2.join()
    bispectrum.join()


    # # DATA STORAGE SECTION # #
    # Executed only once the test has finished.
    print(Fore.RED + 'Test finished successfully, storing data now...' + Style.RESET_ALL)
    print(Fore.GREEN + 'Data has been stored successfully. Preparing bispectrum results...' + Style.RESET_ALL)

    if experiment_phase == "calibration":
        df_norm = np.zeros((bispectrum_length, N_CH*N_CH))
        #Create dataframes to estimate the eyes open mean matrix
        sum = pd.read_csv(f'{folder}/Bispectrum/Calibration_data.csv', index_col=0, usecols=lambda x: x != 'Timestamp')
        arrange = sum.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True)
        arrange.to_csv(f'{folder}/Bispectrum/Nested_loops.csv')


        eyes_open = pd.read_csv(f'{folder}/Bispectrum/Nested_loops.csv', index_col=0)
        df_eo = pd.DataFrame(eyes_open).rename(columns={f'COMB{i+1}': i for i in range(df_norm.shape[1])})
        divisor = len(df_eo)/bispectrum_length
        dic_eo = df_eo.to_dict('dict')


        # Create an array to store the relevant keys
        relevant_keys = np.arange(0, len(df_eo), bispectrum_length)
        
        #Nested loops to sum the relevant values for each combination and store the mean bispectrum in a dataframe
        for i in range(bispectrum_length):
            for comb, bis in dic_eo.items():
                # Calculate the indices to access values in bis
                indices = relevant_keys + i
                # Sum the relevant values using NumPy's array operations
                sum_values = np.sum([bis[key] for key in indices])
                df_norm[i, int(comb)] = sum_values / divisor
        comb_cols = [f'COMB{i}' for i in range(1, df_norm.shape[1] + 1)]
        pd.DataFrame(df_norm, columns=comb_cols).to_csv(f'{folder}/Bispectrum/Mean.csv', index=True, index_label='')
        print(Fore.GREEN + f'Mean bispectrum has been stored successfully in {folder}/Bispectrum/Mean.csv.' + Style.RESET_ALL)

    elif experiment_phase == "interaction":

        # # POST REAL-TIME BISPECTRUM PLOTS SECTION # #
        #Create dataframes for bispectrum results
        data_meanb = pd.read_csv(f'{folder}/Bispectrum/Frequency_bands_bispectrum.csv', index_col=0)
        data_graph = data_meanb.apply(pd.to_numeric, errors='coerce').dropna(axis=0).reset_index(drop=True)
                            
        df_graph = pd.DataFrame(data_graph)
        # Replace -inf and inf with 0 in your DataFrame
        data_graph = data_graph.replace([float('-inf'), float('inf')], 0)
        print(df_graph)
        # Generate a time index from 0 to test_duration seconds with the same length as your DataFrame
        time_index = np.linspace(0, test_duration, len(data_graph))
        for column in data_graph.columns:
            plt.figure(figsize=(8, 6))
            plt.plot(time_index, data_graph[column], label=column)
            plt.title(f'Plot of {column}')
            plt.xlabel('Time (s)')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{folder}/Figures/{column}_plot.png')

####### Sources ########
# To understand the methods and structure of this code, visit the following sources:
# - BrainFlow documentation: https://brainflow.readthedocs.io/en/stable/
# - PyQt5 documentation: https://www.riverbankcomputing.com/static/Docs/PyQt5/
# - Python multiprocessing documentation: https://docs.python.org/3/library/multiprocessing.html
# - Published article: https://doi.org/10.3390/s24061776

