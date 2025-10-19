import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations
import time

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

    try:
        while (True):
            time.sleep(4)
            data = board.get_board_data()  # get latest 256 packages or less, doesn't remove them from internal buffer.

            ############## Data collection #################
            # Empty DataFrames are created for raw data.
            df_signal = pd.DataFrame(columns=['MV' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            df_crudas = pd.DataFrame(columns=['CH' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            
            # The total number of EEG channels is looped to obtain MV for each channel, and
            # thus saved it on the corresponding columns of the respective DataFrame.
            for eeg_channel in eeg_channels:
                df_crudas['CH' + str(eeg_channel)] = data[eeg_channel]
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
            
            # Calculate the new variable based on the formula
            referenced_electrodes = pd.DataFrame()
            referenced_electrodes['referenced_electrode1'] = df_signal['MV3'] - ((df_signal['MV1'] + df_signal['MV2']) / 2)
            referenced_electrodes['referenced_electrode2'] = df_signal['MV4'] - ((df_signal['MV1'] + df_signal['MV2']) / 2)

            # Both raw and PSD DataFrame is exported as a CSV.
            arrange=referenced_electrodes.to_dict('dict')
            lista1 = list(arrange['referenced_electrode1'].values())
            lista2 = list(arrange['referenced_electrode2'].values())
            
            datach1[:800] = lista1[:800]
            datach2[:800] = lista2[:800]


            with second.get_lock():
                # When seconds reach the value, we exit the functions.
                if(second.value == 21):
                    return      

    except KeyboardInterrupt:
        board.stop_stream()
        board.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- End the session with Enophones {device_name} ---')