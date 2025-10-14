import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations
import time
from singal_plotting_all import Graph

# # CODE FOR EEG # #
def EEG(second, folder, datach1, datach2, mac_address, device_name, board_id):
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
    graph = Graph(eeg_channels, sampling_rate)

    try:
        while (True):
            time.sleep(4)
            data = board.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.

            ############## Data collection #################
            # Empty DataFrames are created for raw data.
            df_crudas = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            signal = pd.DataFrame(columns=['CH' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            
            # The total number of EEG channels is looped to obtain MV for each channel, and
            # thus saved it on the corresponding columns of the respective DataFrame.
            for eeg_channel in eeg_channels:
                signal['CH' + str(eeg_channel)] = data[eeg_channel]
                DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
                ####################START OF PREPROCESING#############################
                #Filter for envirionmental noise (Notch: 0=50Hz 1=60Hz)
                DataFilter.remove_environmental_noise(data[eeg_channel], sampling_rate, noise_type=1)
                #Bandpass Filter
                DataFilter.perform_lowpass(data[eeg_channel], sampling_rate, cutoff=100, order=4, filter_type=0, ripple=0)
                DataFilter.perform_highpass(data[eeg_channel], sampling_rate, cutoff=0.1, order=4, filter_type=0, ripple=0)      
                df_crudas['MV' + str(eeg_channel)] = data[eeg_channel]
            
            signal.to_csv(f'{folder}/Raw/{device_name}_Testing.csv', mode='a')
            df_crudas.to_csv(f'{folder}/Raw/{device_name}_Crudas.csv', mode='a')
            
            # Send data to the graph in the background if it is still open
            if graph.running:
                graph.data_signal.emit(data)
                graph.processed_data.emit(df_crudas.values)

            # Calculate the new variable based on the formula
            referenced_electrodes = pd.DataFrame()
            referenced_electrodes['referenced_electrode1'] = df_crudas['MV3'] - ((df_crudas['MV1'] + df_crudas['MV2']) / 2)
            referenced_electrodes['referenced_electrode2'] = df_crudas['MV4'] - ((df_crudas['MV1'] + df_crudas['MV2']) / 2)

            # Both raw and PSD DataFrame is exported as a CSV.
            arrange=referenced_electrodes.to_dict('dict')
            lista1 = list(arrange['referenced_electrode1'].values())
            lista2 = list(arrange['referenced_electrode2'].values())
            
            datach1[:800] = lista1[:800]
            datach2[:800] = lista2[:800]


            with second.get_lock():
                # When seconds reach the value, we exit the functions.
                if(second.value == 21):
                    BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- End the session with {device_name} ---')
                    graph.close_signal.emit()
                    break  # exit loop

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- Interrupted, ending session with {device_name} ---')
        graph.close_signal.emit()

    finally:
        try:
            board.stop_stream()
        except Exception:
            pass  # already stopped
        board.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- Session released for {device_name} ---')