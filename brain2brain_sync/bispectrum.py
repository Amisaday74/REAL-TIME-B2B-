import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, LogLevels

# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.
def bispec(eno1_buffer, eno1_write_idx, eno1_lock, eno2_buffer, eno2_write_idx, eno2_lock, second, folder, event1, event2, completion_event, bispectrum_channels, experiment_phase):
    def read_ring(buffer, write_idx, lock, window_size):
        """
        Returns last `window_size` samples as a stable copy
        Shape: [channels, window_size]
        """
        buffer_len = buffer.shape[1]

        with lock:
            idx = write_idx.value
            start = (idx - window_size) % buffer_len

            if start < idx:
                data = buffer[:, start:idx]
                print(f"Reading data from ring buffer from index {start} to {idx}: {data}")
            else:
                data = np.hstack((
                    buffer[:, start:],
                    buffer[:, :idx]
                ))

            # same safe reading pattern as reader()
            data = data.copy()

        return data
    
    WINDOW_SAMPLES = 1000
    N_CH = bispectrum_channels
    bispectrum_length = WINDOW_SAMPLES // 2# Get the first (and only) element from the lis

    try:
        while (True):
            # Wait for both EEG devices to have data ready
            event1.wait()
            event2.wait()

            # Read last 4 seconds of data from both devices
            # NumPy views (VERY IMPORTANT)
            eno1_buffer_np = np.frombuffer(eno1_buffer, dtype=np.float64).reshape(N_CH, -1)

            eno2_buffer_np = np.frombuffer(eno2_buffer, dtype=np.float64).reshape(N_CH, -1)
            eno1_window = read_ring(eno1_buffer_np, eno1_write_idx, eno1_lock, WINDOW_SAMPLES)

            eno2_window = read_ring(eno2_buffer_np, eno2_write_idx, eno2_lock, WINDOW_SAMPLES)

            print(f"Memory ring buffer after read: {eno1_window}")

            # Shape: [samples, channels]
            matrix_eno1t = eno1_window.T
            matrix_eno2t = eno2_window.T
            print(f"Showing data before bispectrum calculus{matrix_eno1t}")


            cont = 0

            B = np.zeros((N_CH*N_CH, bispectrum_length))  # Initialize B with the correct shape
            index = np.zeros((N_CH*N_CH, 2))
            

            for ch2 in range(N_CH):
                for ch1 in range(N_CH):
                    bs = np.abs(np.fft.fft(matrix_eno1t[:, ch1])*np.fft.fft(matrix_eno2t[:, ch2])*np.conj(np.fft.fft(matrix_eno1t[:, ch1]+matrix_eno2t[:, ch2])))
                    B[cont, :] = np.log(bs[:len(bs)//2].T)  # Mean windows bs on all channels
                    index[cont, :] = [ch1+1, ch2+1]  # Indexing combination order: ch1,ch2
                    cont += 1
            print(B)
            
            
            bispectrum = pd.DataFrame(B)
            b_transpose = bispectrum.transpose()


            df_bispec = pd.DataFrame(columns=['COMB' + str(channel) for channel in range(0, len(index))])
            for eeg_channel2 in range (0,4):
                df_bispec['COMB' + str(eeg_channel2)] = b_transpose[eeg_channel2]
            print(df_bispec)
            
            inspection = df_bispec.copy()
            # Add timestamp column
            current_second = None
            with second.get_lock():
                current_second = second.value
            inspection['Timestamp'] = current_second

            if experiment_phase == "calibration":
                inspection.to_csv(f'{folder}/Bispectrum/Bispec_inspection.csv', mode='a')
                df_bispec.to_csv(f'{folder}/Bispectrum/Calibration_data.csv', mode='a')
            
            if experiment_phase == "interaction":
                inspection.to_csv(f'{folder}/Bispectrum/Bispec_inspection.csv', mode='a')
                df_bispec.to_csv(f'{folder}/Bispectrum/Interaction_data.csv', mode='a')
                df_mean = pd.read_csv(f'{folder}/../Calibration_data/Bispectrum/mean.csv', index_col=0)
                print(df_mean)
                df_sub = df_bispec.sub(df_mean)
                df_div = df_sub.div(df_mean)
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

                bispectrum_mean.to_csv('{}/Frequency_bands_bispectrum.csv'.format(folder), mode='a')
                print(bispectrum_mean)

            # Clear events for next iteration
            event1.clear()
            event2.clear()
            # Check if completion_event has been set by stopwatch (master controller)
            if completion_event.is_set():
                return

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- Ending bispectrum analysis ---')