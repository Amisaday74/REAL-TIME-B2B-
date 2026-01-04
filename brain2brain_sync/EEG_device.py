import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations
import time
import numpy as np

#-------------------- CODE FOR EEG DEVICE CONNECTION AND DATA PROCESSING --------------------#
def EEG(second, folder, buffer_np, write_idx, lock, mac_address, device_name, board_id, queue, event):

    def write_ring(buffer, write_idx, lock, new_data):
        """
        buffer: np.ndarray [channels, buffer_len]
        new_data: np.ndarray [channels, n_samples]
        """
        n_samples = new_data.shape[1]
        buffer_len = buffer.shape[1]

        with lock:
            idx = write_idx.value
            end = idx + n_samples

            if end <= buffer_len:
                buffer[:, idx:end] = new_data
                print(f"Writing data to ring buffer from index {idx} to {end}: {buffer}")
            else:
                first = buffer_len - idx
                buffer[:, idx:] = new_data[:, :first]
                buffer[:, :end % buffer_len] = new_data[:, first:]
                print(f"Writing data to ring buffer with wrap-around from index {idx} to {end % buffer_len}: {buffer}")

            write_idx.value = end % buffer_len
            return buffer.copy()

    # The following object will save parameters to connect with the EEG device.
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    # If the device is Enophone, the MAC address has to be specified.
    if board_id == BoardIds.ENOPHONE_BOARD.value:
        params.mac_address = mac_address

    # Relevant variables are obtained from the EEG device connected.
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)
    board = BoardShim(board_id, params)

    # Define constants for data collection
    time_window = 4
    data_samples = 1000
    last_window = -1  

    ############# Session is then initialized #######################
    board.prepare_session()
    board.start_stream(45000, f"file://{folder}/{device_name}_testOpenBCI.csv:w")
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- Starting the streaming with {device_name} ---')

    try:
        while (True):
            # ---------- TIME WINDOWS LOGIC ----------
            """
            --------- FIRST 4-SECOND WINDOW (WINDOW 0)---------
            Real time (s)	second.value
            0.00 - 0.99	          0
            1.00 - 1.99	          1
            2.00 - 2.99	          2
            3.00 - 3.99	          3     
            --------- NEXT 4-SECOND WINDOW (WINDOW 1)---------
            4.00 - 4.99	          4
            """
            with second.get_lock():
                current_second = second.value  # Make this function responsive to main timer updates
                if(current_second >= 20):
                    BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- End the session with {device_name} ---')
                    break  # exit loop

            current_window = current_second // time_window # Determine the current 4-second window 
            if current_window == last_window:
                time.sleep(0.01) # Cooperative scheduling pause, serves to prevent busy-waiting, reducing CPU usage and jitter for real-time systems.
                continue
            last_window = current_window

            # ---------- DATA ACQUISITION ----------
            # Loop to start collecting data, processing happens only when sample count is sufficient, not merely when time has passed.
            while True:
                data = board.get_current_board_data(data_samples)  #Use get_current_board_data(n) to get latest collected data and don't remove it from internal buffer.
                if data.shape[1] >= data_samples:
                    break
                time.sleep(0.05) # Allows driver threads to run safely and prevents buffers trashing.

            data = board.get_board_data(data_samples) #get all collected data and flush it from internal buffer
            raw_data = data.copy() # make a copy of raw data for storage and graphs avoiding racing conditions

            # ---------- PREPROCESSING AND DATAFRAME CREATION ----------
            df_signal = pd.DataFrame(columns=['CH' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            df_raw = pd.DataFrame(columns=['CH' + str(channel) for channel in range(1, len(eeg_channels) + 1)])
            
            # The total number of EEG channels is looped to process EEG data (uV) for each channel
            for eeg_channel in eeg_channels:
                df_raw['CH' + str(eeg_channel)] = data[eeg_channel]
                DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
                #Filter for envirionmental noise (Notch: 0=50Hz 1=60Hz)
                DataFilter.remove_environmental_noise(data[eeg_channel], sampling_rate, noise_type=1)
                #Bandpass Filter
                DataFilter.perform_lowpass(data[eeg_channel], sampling_rate, cutoff=100, order=4, filter_type=0, ripple=0)
                DataFilter.perform_highpass(data[eeg_channel], sampling_rate, cutoff=0.1, order=4, filter_type=0, ripple=0)      
                df_signal['CH' + str(eeg_channel)] = data[eeg_channel]
            
            df_raw["Timestamp (UNIX)"] = data[timestamp_channel]
            df_raw.to_csv(f'{folder}/Real_Time_Data/{device_name}_raw_data.csv', mode='a')
            df_signal.to_csv(f'{folder}/Real_Time_Data/{device_name}_signal_processing.csv', mode='a')

            # Send data to the graph in the background if it is still open
            queue.put((device_name, raw_data, data))
            
            # Calculate the new variable based on the formula
            referenced_electrodes = pd.DataFrame()
            referenced_electrodes['referenced_electrode1'] = df_signal['CH3'] - ((df_signal['CH1'] + df_signal['CH2']) / 2)
            referenced_electrodes['referenced_electrode2'] = df_signal['CH4'] - ((df_signal['CH1'] + df_signal['CH2']) / 2)

            # Write to ring buffer
            ring_block = np.vstack([
                referenced_electrodes['referenced_electrode1'].values,
                referenced_electrodes['referenced_electrode2'].values
            ])

            buffer_np = write_ring(buffer_np, write_idx, lock, ring_block)
            print(f"Memory ring buffer after write: {buffer_np} ")

            # Signal that data is ready
            event.set()


    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- Interrupted, ending session with {device_name} ---')

    finally:
        board.stop_stream()
        board.release_session()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, f' ---- Session released for {device_name} ---')