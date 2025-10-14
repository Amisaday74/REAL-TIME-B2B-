import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_rel

directory_path = r'E:\Empatica-Project-ALAS-main-20230212T023903Z-001\Pruebas estadisticas'

# List of frequency bands
frequency_bands = ['alpha', 'beta', 'delta', 'theta', 'gamma']

# Iterate through frequency bands
for band in frequency_bands:
    # Construct the full paths to the CSV files
    csv_file_name = f'stacked_{band}_S371R02_28052023_1841.csv'
    csv_file_name2 = f'stacked_{band}_S310R02_28052023_1623.csv'
    
    csv_file_path = os.path.join(directory_path, csv_file_name)
    csv_file_path2 = os.path.join(directory_path, csv_file_name2)

    # Read the CSV files into pandas DataFrames with default column names
    df = pd.read_csv(csv_file_path, header=None)
    df2 = pd.read_csv(csv_file_path2, header=None)

    # Find the minimum number of rows
    min_rows = min(len(df), len(df2))

    # Slice both DataFrames to the minimum number of rows
    df = df.head(min_rows)
    df2 = df2.head(min_rows)

    # Perform t-test column-wise between df and df2
    results = []

    for col_idx in range(len(df.columns)):
        t_statistic, p_value = ttest_rel(df.iloc[:, col_idx], df2.iloc[:, col_idx])
        results.append({'Column': f'{col_idx}', 'T-Statistic': t_statistic, 'P-Value': p_value})

    # Create a DataFrame from the results
    t_test_results = pd.DataFrame(results)

    # Print the t-test results for the current frequency band
    print(f'Test results for {band} band:')
    print(t_test_results)
