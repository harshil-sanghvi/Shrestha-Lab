import os
import pandas as pd
import numpy as np
from pandas import ExcelWriter
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

def check_saa_in_path(path):
    """Check if 'SAA' is present in the path."""
    return 'SAA' in path.split('\\')[-2].upper()

def process_file(file_path, total_trials, col):
    """Process an individual file and extract required data."""
    try:
        df = pd.read_csv(file_path, header=None).fillna('00:00.0').iloc[:, 1:5]
        df.columns = ['Time', 'Event', 'C', 'Desc']

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

        # Calculate counts
        av_count = df[df['Desc'] == 'Avoidance'].shape[0] // 2
        esc_count = df[df['Desc'] == 'Escape'].shape[0] // 2
        fail_count = total_trials - av_count - esc_count

        # Calculate percentages
        av_perc = round(av_count / total_trials * 100, 1)
        esc_perc = round(esc_count / total_trials * 100, 1)
        fail_perc = abs(round(100 - av_perc - esc_perc, 1))

        animal_id = os.path.basename(file_path).split('_')[-1].split('.')[0]
        
        temp_df = pd.DataFrame(np.reshape([animal_id, av_count, esc_count, fail_count, av_perc, esc_perc, fail_perc, entry_count] + entry_cs + exit_cs + latency_cs, (1, -1)), columns=col)
        return temp_df
    except Exception as e:
        print(file_path.split('\\')[-4].split()[-1], file_path.split('\\')[-3], animal_id, " -> ", e)
        return None

def process_gs_data(GS_DIR_PATH):
    """Process data from files in GS_DIR_PATH."""
    # Check if GS_DIR_PATH contains 'SAA'
    total_trials = 11 if check_saa_in_path(GS_DIR_PATH) else 5

    # Define column names for the DataFrame
    col = pd.MultiIndex.from_arrays([
        ['Animal ID'] + ['SAA#']*3 + ['SAA%']*3 + ['n[Shuttle]'] + ['entry']*total_trials +
        ['exit']*total_trials + ['latency']*total_trials,
        [''] + ['Av', 'Esc', 'Fail'] * 2 + [''] + ['CS' + str(i) for i in range(1, total_trials + 1)] * 2 +
        ['CS' + str(i) for i in range(1, total_trials + 1)]
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

def add_animal_details(data_df, exp_df, ct):
    """Add animal details from exp_df to data_df based on common identifiers."""
    na_mask = exp_df['SN'].isna() | exp_df['SN'].isnull()
    start_index = exp_df[~na_mask & exp_df['SN'].str.endswith(ct)].index[0]

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
    
    final_df = pd.concat([exp_df_renamed, data_df], axis=1, join='inner', sort=False)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    return final_df

def align_center(x):
    """Align text in cells to center."""
    return ['text-align: center' for _ in x]

def process_and_save_data(PATH, exp_df, ct, add_animal_info=True):
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
                data_df = add_animal_details(data_df, exp_df, ct)
                data_df.set_index('SN', inplace=True)
            data_df.sort_index(inplace=True)
            data_df.style.apply(align_center, axis=0).to_excel(writer, sheet_name=subfolder, index=True)

    writer.close()

PATH = r'path/to/data' # Path to the data folder
EXP_DETAILS_PATH = r'path/to/details.xlsx' # Path to the Excel file containing experiment details

exp_df = pd.read_excel(EXP_DETAILS_PATH, usecols=[0, 1, 2, 3, 4])
exp_df.columns = ['SN', 'Animal', 'Sex', 'Subject ID', 'Group ']

for subfolder in tqdm(sorted(os.listdir(PATH)), desc="Processing subfolders", unit="folder"):
    ct = subfolder.split()[-1]
    GS_DIR_PATH = os.path.join(PATH, subfolder)
    try:
        if os.path.isdir(GS_DIR_PATH):
            process_and_save_data(GS_DIR_PATH, exp_df, ct)
    except Exception as e:
        print(f"Error processing {GS_DIR_PATH}: {e}")