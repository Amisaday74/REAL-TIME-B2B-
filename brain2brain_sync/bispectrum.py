import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, LogLevels

# Finally, for both processes to run, this condition has to be met. Which is met
# if you run the script.
def bispec(eno1_buffer, eno1_write_idx, eno1_lock,
           eno2_buffer, eno2_write_idx, eno2_lock,
           second, folder, event1, event2):
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

        return data.copy()
    
    WINDOW_SAMPLES = 1000
    N_CH = 2

    try:
        while (True):
            # Wait for both EEG devices to have data ready
            event1.wait()
            event2.wait()

            # Read last 4 seconds of data from both devices
            eno1_window = read_ring(
                eno1_buffer, eno1_write_idx, eno1_lock, WINDOW_SAMPLES
            )

            eno2_window = read_ring(
                eno2_buffer, eno2_write_idx, eno2_lock, WINDOW_SAMPLES
            )

            # Shape: [samples, channels]
            matrix_eno1t = eno1_window.T
            matrix_eno2t = eno2_window.T
            print(f"Showing data before bispectrum calculus{matrix_eno1t}")


            cont = 0
            Nch=2

            B = np.zeros((Nch*Nch, WINDOW_SAMPLES//2))
            index = np.zeros((Nch*Nch, 2))
            

            for ch2 in range(Nch):
                for ch1 in range(Nch):
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
            df_norm = np.zeros((len(df_bispec), Nch*Nch))
            
            inspection = df_bispec.copy()
            # Add timestamp column
            current_second = None
            with second.get_lock():
                current_second = second.value
            inspection['Timestamp'] = current_second
            inspection.to_csv('{}/Bispec_inspection.csv'.format(folder), mode='a')
            
            df_bispec.to_csv('{}/Bispec.csv'.format(folder), mode='a')
            

            with second.get_lock():
                # When the seconds reach 312, we exit the functions.
                if(second.value >= 20):
                    return
                elif (second.value <= 8):
                    #Get data to apply normalization
                    for i in range (1):
                        print('Preparing device calibration...')
                        df_eo = df_bispec
                        df_eo.to_csv('{}/Calibration_data.csv'.format(folder), mode='a')

                elif ((second.value > 8) and (second.value <= 20)):
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
                        relevant_keys = np.arange(0, len(df_eo), 500)
                        
                        for i in range(500):
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

            bispectrum_mean.to_csv('{}/Frequency_bands_bispectrum.csv'.format(folder), mode='a')



            print(bispectrum_mean)
            # Clear events for next iteration
            event1.clear()
            event2.clear()

    except KeyboardInterrupt:
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, ' ---- End the session with Enophone 2 ---')