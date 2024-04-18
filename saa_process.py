import os
import pandas as pd
import numpy as np
from pandas import ExcelWriter
import warnings
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")

def check_saa_in_path(path):
    """Check if 'SAA' is present in the path."""
    return 'SAA' in path.split('\\')[-2].upper()

def process_file(file_path, total_trials, col):
    """Process an individual file and extract required data."""
    try:
        df = pd.read_csv(file_path, header=None).fillna('00:00.0').iloc[:, 1:5]
        df.columns = ['Time', 'Event', 'C', 'Desc']
        
        total_duration = df[df['Event'] == 'Finish']['Time'].astype(int).values[0]
        
        # Count number of rows with 'Right Entry' or 'Left Entry' in Desc
        entry_count = df[(df['Desc'] == 'Right Entry') | (df['Desc'] == 'Left Entry')].shape[0]

        # Filter only necessary data
        cs_df = df[(df['Desc'] == 'CS (Tone)')]
        cs_df['Seconds'] = cs_df['Time'].astype(int)

        # Separate entry and exit CS
        entry_cs = cs_df[cs_df['Event'] == 'Entry']['Seconds'].tolist()
        exit_cs = cs_df[cs_df['Event'] == 'Exit']['Seconds'].tolist()

        # Calculate latency
        latency_cs = [exit - entry for exit, entry in zip(exit_cs, entry_cs)]
        
        # calculate total CS duration
        total_cs_duration = [sum(latency_cs)]
        
        # calculate average latency
        avg_latency = [round(sum(latency_cs) / len(latency_cs), 1)]
        
        # Calculate counts
        av_count = df[df['Desc'] == 'Avoidance'].shape[0] // 2
        esc_count = df[df['Desc'] == 'Escape'].shape[0] // 2
        fail_count = total_trials - av_count - esc_count

        # Calculate percentages
        av_perc = round(av_count / total_trials * 100, 1)
        esc_perc = round(esc_count / total_trials * 100, 1)
        fail_perc = abs(round(100 - av_perc - esc_perc, 1))

        entry_count_non_cs = entry_count - av_count

        n_shuttl_per_ten_min_cs = round((av_count/total_cs_duration[0]) * 600, 2)
        n_shuttl_per_ten_min_non_cs = round((entry_count_non_cs/(total_duration - total_cs_duration[0])) * 600, 2)
        
        animal_id = os.path.basename(file_path).split('_')[-1].split('.')[0].split()[-1]

        temp_df = pd.DataFrame(np.reshape([animal_id, av_count, esc_count, fail_count, av_perc, esc_perc, fail_perc, entry_count, entry_count_non_cs, n_shuttl_per_ten_min_cs, n_shuttl_per_ten_min_non_cs, total_duration] + entry_cs + exit_cs + latency_cs + total_cs_duration + avg_latency, (1, -1)), columns=col)
        return temp_df
    except Exception as e:
        print(file_path.split('\\')[-4].split()[-2], file_path.split('\\')[-3], file_path.split('\\')[-1].split('.')[-2]," -> ", e)
        return None

def process_gs_data(GS_DIR_PATH):
    """Process data from files in GS_DIR_PATH."""
    # Check if GS_DIR_PATH contains 'SAA'
    total_trials = 11 if check_saa_in_path(GS_DIR_PATH) else 5

    # Define column names for the DataFrame
    col = pd.MultiIndex.from_arrays([
        ['Animal ID'] + ['SAA#']*3 + ['SAA%']*3 + ['n[Shuttle]']*2 + ['n[Shuttle]/10min']*2 + ['Total'] + ['entry']*total_trials +
        ['exit']*total_trials + ['latency']*total_trials + ['Total CS']  + ['Average'],
        [''] + ['Av', 'Esc', 'Fail'] * 2 + ['Total'] + ['non CS'] + ['CS'] + ['non CS'] + ['Duration'] + ['CS' + str(i) for i in range(1, total_trials + 1)] * 2 +
        ['CS' + str(i) for i in range(1, total_trials + 1)] + ['Duration'] + ['latency']
    ])

    data_df = pd.DataFrame(columns=col)
    file_names = os.listdir(GS_DIR_PATH)

    # Process each file in GS_DIR_PATH
    for file_name in file_names:
        file_path = os.path.join(GS_DIR_PATH, file_name)
        if os.path.isfile(file_path):
            temp_df = process_file(file_path, total_trials, col)
            if temp_df is not None:
                data_df = pd.concat([data_df, temp_df], ignore_index=True)

    return data_df

def add_animal_details(data_df, exp_df, ct, dt):
    """Add animal details from exp_df to data_df based on common identifiers."""
    na_mask = exp_df['SN'].isna() | exp_df['SN'].isnull() # Check for NaN values in SN column
    start_index = exp_df[~na_mask & exp_df['SN'].str.contains(ct) & exp_df['SN'].str.contains(dt)].index[0] # Get the index of the first non-NaN value in SN column

    next_index_with_nan = None
    for index in range(start_index + 1, len(exp_df)):
        if exp_df.iloc[index].isnull().all():
            next_index_with_nan = index
            break

    exp_df = exp_df[start_index+2:next_index_with_nan]
    exp_df.reset_index(drop=True, inplace=True)
    exp_df.sort_values(by='Animal', inplace=True)
    exp_df.columns = pd.MultiIndex.from_arrays([exp_df.columns, ['']*len(exp_df.columns)])
    exp_df_renamed = exp_df.rename(columns={'Animal': 'Animal ID'})
    
    final_df = pd.merge(exp_df_renamed, data_df, on='Animal ID', how='inner')
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    return final_df

def align_center(x):
    """Align text in cells to center."""
    return ['text-align: center' for _ in x]

def process_and_save_data(PATH, exp_df, ct, dt, add_animal_info=True):
    """Process data in subfolders of PATH and save to Excel."""
    title_split = PATH.split('\\')
    info = title_split[-1].split()
    title = info[1].split('_')[0] + ' ' + (info[1].split('_')[1]).upper() + ' ' + info[2] + ' ' + info[0]
    output_file = title + '.xlsx'
    writer = ExcelWriter(output_file)

    for subfolder in sorted(os.listdir(PATH)):
        GS_DIR_PATH = os.path.join(PATH, subfolder)
        if os.path.isdir(GS_DIR_PATH) and ('SAA' in subfolder.upper() or 'LTM' in subfolder.upper()):
            GS_DIR_PATH = os.path.join(GS_DIR_PATH, 'csv files')
            data_df = process_gs_data(GS_DIR_PATH)  # Assuming process_data is defined elsewhere
            if add_animal_info:
                data_df = add_animal_details(data_df, exp_df, ct, dt)
                data_df.set_index('SN', inplace=True)
            data_df.sort_index(inplace=True)
            data_df.style.apply(align_center, axis=0).to_excel(writer, sheet_name=subfolder, index=True)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process data from subfolders in a folder and save to Excel.")
    parser.add_argument("--path", help="Path to the data folder", type=str, required=True)
    parser.add_argument("--exp_details_path", help="Path to the Excel file containing experiment details", type=str, required=True)
    args = parser.parse_args()

    exp_df = pd.read_excel(args.exp_details_path, usecols=[0, 1, 2, 3, 4])
    exp_df.columns = ['SN', 'Animal', 'Sex', 'Subject ID', 'Group ']

    for subfolder in tqdm(sorted(os.listdir(args.path)), desc="Processing subfolders", unit="folder"):
        ct = subfolder.split()[-2] # if using old WT SAA data, use subfolder.split()[-1]. For newer data following established naming convention, use subfolder.split()[-2]
        dt = subfolder.split()[0]
        GS_DIR_PATH = os.path.join(args.path, subfolder)
        try:
            if os.path.isdir(GS_DIR_PATH):
                process_and_save_data(GS_DIR_PATH, exp_df, ct, dt)
        except Exception as e:
            print(f"Error processing {GS_DIR_PATH}: {e}")