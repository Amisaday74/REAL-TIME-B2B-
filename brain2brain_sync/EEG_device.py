import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations
import time

# # CODE FOR EEG # #
def EEG(second, folder, datach1, datach2, mac_address, device_name, board_id, queue, event):
    # The following object will save parameters to connect with the EEG.
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    if board_id == BoardIds.ENOPHONE_BOARD.value:
        params.mac_address = mac_address

    # Relevant variables are obtained from the current EEG device connected.
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)


    ############# Session is then initialized #######################
    board.prepare_session()
    board.start_stream(45000, f"file://{folder}/{device_name}_testOpenBCI.csv:w")
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- Starting the streaming with {device_name} ---')

    # Initialize the Graph in a separate thread
    # graph = Graph(eeg_channels, sampling_rate)

    try:
        while (True):
            time.sleep(4)
            data = board.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.
            if data.shape[1] < 1000:
                print(f"Data packet too small: {data.shape[1]} samples. Attempting to recover...")
                time.sleep(0.1)
                continue

            # make a copy of raw data for storage and graphs avoiding racing conditions
            raw_data = data.copy()

            ############## Data collection #################
            # Empty DataFrames are created for raw data.
            df_signal = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            df_crudas = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            
            # The total number of EEG channels is looped to obtain MV for each channel, and
            # thus saved it on the corresponding columns of the respective DataFrame.
            for eeg_channel in eeg_channels:
                df_crudas['MV' + str(eeg_channel)] = data[eeg_channel]
                DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
                ####################START OF PREPROCESING#############################
                #Filter for envirionmental noise (Notch: 0=50Hz 1=60Hz)
                DataFilter.remove_environmental_noise(data[eeg_channel], sampling_rate, noise_type=1)
                #Bandpass Filter
                DataFilter.perform_lowpass(data[eeg_channel], sampling_rate, cutoff=100, order=4, filter_type=0, ripple=0)
                DataFilter.perform_highpass(data[eeg_channel], sampling_rate, cutoff=0.1, order=4, filter_type=0, ripple=0)      
                df_signal['MV' + str(eeg_channel)] = data[eeg_channel]
            
            df_crudas.to_csv(f'{folder}/Real_Time_Data/{device_name}_raw_data.csv', mode='a')
            df_signal.to_csv(f'{folder}/Real_Time_Data/{device_name}_signal_processing.csv', mode='a')

            # Send data to the graph in the background if it is still open
            queue.put((device_name, raw_data, data))
            
            # Calculate the new variable based on the formula
            referenced_electrodes = pd.DataFrame()
            referenced_electrodes['referenced_electrode1'] = df_signal['MV3'] - ((df_signal['MV1'] + df_signal['MV2']) / 2)
            referenced_electrodes['referenced_electrode2'] = df_signal['MV4'] - ((df_signal['MV1'] + df_signal['MV2']) / 2)

            # Both raw and PSD DataFrame is exported as a CSV.
            arrange=referenced_electrodes.to_dict('dict')
            lista1 = list(arrange['referenced_electrode1'].values())
            lista2 = list(arrange['referenced_electrode2'].values())
            
            datach1[:1000] = lista1[:1000]
            datach2[:1000] = lista2[:1000]

            # Signal that data is ready
            event.set()


            with second.get_lock():
                # When seconds reach the value, we exit the functions.
                if(second.value >= 20):
                    BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- End the session with {device_name} ---')
                    break  # exit loop

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- Interrupted, ending session with {device_name} ---')

    finally:
        board.stop_stream()
        board.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- Session released for {device_name} ---')