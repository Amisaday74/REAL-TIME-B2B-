import os
import pandas as pd
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the directory path
directory_path = r'E:\Empatica-Project-ALAS-main-20230212T023903Z-001\S360R01_28052023_1727\Raw'
directory_path_s2 = r'E:\Empatica-Project-ALAS-main-20230212T023903Z-001\S360R01_28052023_1727\Raw 2'

# Specify the CSV file name
csv_file_name = 'crudas.csv'
csv_file_name_s2 = 'crudas2.csv'

# Construct the full path to the CSV file
csv_file_path = os.path.join(directory_path, csv_file_name)
csv_file_path_s2 = os.path.join(directory_path_s2, csv_file_name_s2)

# Initialize a list to store the starting row indices
start_indices = []
start_indices2 = []

# Read the CSV file line by line to find the rows with column names
with open(csv_file_path, 'r') as file:
    for idx, line in enumerate(file):
        if 'MV1' in line or 'MV2' in line or 'MV3' in line:
            start_indices.append(idx)

with open(csv_file_path_s2, 'r') as file:
    for idx, line in enumerate(file):
        if 'MV1' in line or 'MV2' in line or 'MV3' in line:
            start_indices2.append(idx)

#ACCESO A ARCHIVOS EN ETAPA DE CALIBRACIÓN
# Define the directory path
directory_path2 = r'E:\Empatica-Project-ALAS-main-20230212T023903Z-001\S360R01_28052023_1727'
#directory_path2 = r'E:\Empatica-Project-ALAS-main-20230212T023903Z-001\S371R01_28052023_1855'
# Specify the CSV file name
csv_file_name2 = 'Calibration_data_clean.csv'
# Construct the full path to the CSV file
csv_file_path2 = os.path.join(directory_path2, csv_file_name2)
# Open the CSV file to count the number of rows
with open(csv_file_path2, 'r') as file:
    num_rows = sum(1 for line in file)-1




# Initialize an empty DataFrame to store the extracted data
extracted_data = pd.DataFrame()
extracted_data2 = pd.DataFrame()

# Read the CSV file and extract the next ten rows after each detected start index
for start_idx in start_indices:
    data = pd.read_csv(csv_file_path, skiprows=range(1, start_idx + 1), nrows=800)
    extracted_data = extracted_data.append(data, ignore_index=True)

for start_idx in start_indices2:
    data2 = pd.read_csv(csv_file_path_s2, skiprows=range(1, start_idx + 1), nrows=800)
    extracted_data2 = extracted_data2.append(data2, ignore_index=True)

# Calculate the new variable based on the formula
extracted_data['MV1'] = pd.to_numeric(extracted_data['MV1'], errors='coerce')
extracted_data['MV2'] = pd.to_numeric(extracted_data['MV2'], errors='coerce')
extracted_data['MV3'] = pd.to_numeric(extracted_data['MV3'], errors='coerce')
extracted_data['MV4'] = pd.to_numeric(extracted_data['MV4'], errors='coerce')
extracted_data['referenced_electrode1'] = extracted_data2['MV3'] - ((extracted_data2['MV1'] + extracted_data2['MV2']) / 2)
extracted_data['referenced_electrode2'] = extracted_data['MV4'] - ((extracted_data2['MV1'] + extracted_data2['MV2']) / 2)

extracted_data2['MV1'] = pd.to_numeric(extracted_data2['MV1'], errors='coerce')
extracted_data2['MV2'] = pd.to_numeric(extracted_data2['MV2'], errors='coerce')
extracted_data2['MV3'] = pd.to_numeric(extracted_data2['MV3'], errors='coerce')
extracted_data2['MV4'] = pd.to_numeric(extracted_data2['MV4'], errors='coerce')


extracted_data2['referenced_electrode1'] = extracted_data2['MV3'] - ((extracted_data2['MV1'] + extracted_data2['MV2']) / 2)
extracted_data2['referenced_electrode2'] = extracted_data2['MV4'] - ((extracted_data2['MV1'] + extracted_data2['MV2']) / 2)

# Create a new DataFrame containing only the reference electrode values
reference_electrodes_df = extracted_data[['referenced_electrode1', 'referenced_electrode2']]
reference_electrodes_df2 = extracted_data2[['referenced_electrode1', 'referenced_electrode2']]

# Optionally, save the reference electrode data to a new CSV file
#output_csv_file = 'reference_electrodes.csv'
#output_csv_path = os.path.join(directory_path, output_csv_file)
#reference_electrodes_df.to_csv(output_csv_path, index=False)

# Print the new DataFrame containing reference electrode values
print(reference_electrodes_df)
print(reference_electrodes_df2)

# Initialize matrices
Nch = 2  # Update this value with the actual number of channels
window_size = 800
num_windows = len(reference_electrodes_df) // window_size

B = np.zeros((num_windows, Nch * Nch, 400))  # Number of windows, Nch * Nch, half of the FFT matrix
index = np.zeros((Nch * Nch, 2))

# Calculate FFT-based values and populate matrices
for window_idx in range(num_windows):
    window_start = window_idx * window_size
    window_end = window_start + window_size
    window_data = reference_electrodes_df.iloc[window_start:window_end]
    window_data2 = reference_electrodes_df2.iloc[window_start:window_end]
    
    cont = 0  # Reset counter for each window
    for ch2 in range(Nch):
        for ch1 in range(Nch):
            bs = np.abs(
                np.fft.fft(window_data.iloc[:, ch1])
                * np.fft.fft(window_data2.iloc[:, ch2])
                * np.conj(np.fft.fft(window_data.iloc[:, ch1] + window_data2.iloc[:, ch2]))
            )
            B[window_idx, cont, :] = np.log(bs[:len(bs) // 2].T)
            index[cont, :] = [ch1 + 1, ch2 + 1]
            cont += 1

# Create a 2D DataFrame by stacking the bispectrum values
stacked_bispectrum = pd.DataFrame(B[1,:,:]).transpose()

num_windows_to_iterate=int(num_rows/len(stacked_bispectrum))

# Initialize a matrix to store the sum of matrices
sum_matrix = np.zeros((Nch * Nch, 400))

for window_idx in range(num_windows_to_iterate):
    sum_matrix += B[window_idx, :, :]

# Calculate the mean matrix
mean_matrix = sum_matrix / num_windows_to_iterate

# Print the mean matrix
print("Mean Matrix:")
print(mean_matrix)



# Create an empty DataFrame to store division results
division_results_df = pd.DataFrame()

# Iterate through the remaining windows
for window_idx in range(num_windows_to_iterate+1, num_windows):
    current_matrix = B[window_idx, :, :]
    subtracted_matrix = current_matrix - mean_matrix
    division_result = subtracted_matrix / mean_matrix
    division_results_df = pd.concat([division_results_df, pd.DataFrame(division_result)], axis=1)

# Transpose the division results DataFrame
division_results_df_transposed = division_results_df.transpose()
print(division_results_df_transposed)

# Define frequency limits
delta_limit = (4 * len(stacked_bispectrum)) // 125  # 125Hz is the frequency limit to the bispectrum matrix length data
theta_limit = (8 * len(stacked_bispectrum)) // 125
alpha_limit = (13 * len(stacked_bispectrum)) // 125
beta_limit = (29 * len(stacked_bispectrum)) // 125
gamma_limit = (50 * len(stacked_bispectrum)) // 125

# Initialize lists for each frequency band
delta_matrices = []
theta_matrices = []
alpha_matrices = []
beta_matrices = []
gamma_matrices = []

# Iterate through the windows and apply operations for each frequency band
for window_idx in range(num_windows_to_iterate+1, num_windows):
    current_matrix = B[window_idx, :, :]
    subtracted_matrix = current_matrix - mean_matrix
    division_result = subtracted_matrix / mean_matrix

    # Divide the result into frequency bands and store in the corresponding lists
    delta_matrices.append(division_result[:, :delta_limit])
    theta_matrices.append(division_result[:, delta_limit:theta_limit])
    alpha_matrices.append(division_result[:, theta_limit:alpha_limit])
    beta_matrices.append(division_result[:, alpha_limit:beta_limit])
    gamma_matrices.append(division_result[:, beta_limit:gamma_limit])

# Assuming you have NumPy arrays for each frequency band
delta_matrices = np.array(delta_matrices)
theta_matrices = np.array(theta_matrices)
alpha_matrices = np.array(alpha_matrices)
beta_matrices = np.array(beta_matrices)
gamma_matrices = np.array(gamma_matrices)


# Transpose the matrices before stacking
transposed_delta = delta_matrices.transpose((0, 2, 1)).reshape(-1, delta_matrices.shape[1])
transposed_theta = theta_matrices.transpose((0, 2, 1)).reshape(-1, theta_matrices.shape[1])
transposed_alpha = alpha_matrices.transpose((0, 2, 1)).reshape(-1, alpha_matrices.shape[1])
transposed_beta = beta_matrices.transpose((0, 2, 1)).reshape(-1, beta_matrices.shape[1])
transposed_gamma = gamma_matrices.transpose((0, 2, 1)).reshape(-1, gamma_matrices.shape[1])

# Stack the transposed matrices vertically
stacked_delta = np.vstack(transposed_delta)
stacked_theta = np.vstack(transposed_theta)
stacked_alpha = np.vstack(transposed_alpha)
stacked_beta = np.vstack(transposed_beta)
stacked_gamma = np.vstack(transposed_gamma)

# Print the shapes of the stacked arrays
print("Stacked Delta Shape:", stacked_delta.shape)
print("Stacked Theta Shape:", stacked_theta.shape)
print("Stacked Alpha Shape:", stacked_alpha.shape)
print("Stacked Beta Shape:", stacked_beta.shape)
print("Stacked Gamma Shape:", stacked_gamma.shape)

# Convert the stacked transposed matrices to DataFrames
df_delta = pd.DataFrame(stacked_delta)
df_theta = pd.DataFrame(stacked_theta)
df_alpha = pd.DataFrame(stacked_alpha)
df_beta = pd.DataFrame(stacked_beta)
df_gamma = pd.DataFrame(stacked_gamma)
# Specify the directory path
output_directory = r'E:\Empatica-Project-ALAS-main-20230212T023903Z-001\Pruebas estadisticas'

# Save the DataFrames to CSV files without column labels
df_delta.to_csv(os.path.join(output_directory, 'stacked_delta_S360R01_28052023_1727.csv'), index=False, header=False)
df_theta.to_csv(os.path.join(output_directory, 'stacked_theta_S360R01_28052023_1727.csv'), index=False, header=False)
df_alpha.to_csv(os.path.join(output_directory, 'stacked_alpha_S360R01_28052023_1727.csv'), index=False, header=False)
df_beta.to_csv(os.path.join(output_directory, 'stacked_beta_S360R01_28052023_1727.csv'), index=False, header=False)
df_gamma.to_csv(os.path.join(output_directory, 'stacked_gamma_S360R01_28052023_1727.csv'), index=False, header=False)


