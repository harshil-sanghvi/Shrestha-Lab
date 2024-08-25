import os
import pandas as pd
import numpy as np
from pandas import ExcelWriter
import warnings
import argparse
import sys
import traceback
from collections import defaultdict

warnings.filterwarnings("ignore")

def check_saa_in_path(path):
    """Check if 'SAA' is present in the path."""
    return 'SAA' in os.path.normpath(path).split(os.sep)[-2].upper()

def process_file(file_path, total_trials, col, time_discrepancy_dict):
    """Process an individual file and extract required data."""
    try:
        # all field except entry and exit should be filled with hyphen
        df = pd.read_csv(file_path, header=None).fillna('00:00.0').iloc[:, 1:5]
        df.columns = ['Time', 'Event', 'C', 'Desc']
        
        total_duration = '-'
        
        # Count number of rows with 'Right Entry' or 'Left Entry' in Desc
        entry_count = '-'

        # Filter only necessary data
        cs_df = df[(df['Desc'] == 'CS (Tone)')]
        cs_df['Seconds'] = cs_df['Time'].astype(int)

        animal_id = os.path.basename(file_path).split('_')[-1].split('.')[0].split()[-1]
        
        ct, exp = file_path.split(os.sep)[-4].split()[-1], file_path.split(os.sep)[-3].upper()

        # Separate entry and exit CS
        entry_cs = cs_df[cs_df['Event'] == 'Entry']['Seconds'].tolist()
        exit_cs = cs_df[cs_df['Event'] == 'Exit']['Seconds'].tolist()

        try:
            # Adjust time discrepancies
            if time_discrepancy_dict[ct][exp][animal_id] != 0:
                entry_cs = [time - time_discrepancy_dict[ct][exp][animal_id] for time in entry_cs]
                exit_cs = [time - time_discrepancy_dict[ct][exp][animal_id] for time in exit_cs]
        except TypeError:
            print(f"Error accessing time discrepancy for {ct} {exp} {animal_id}. Most likely it does not exist.")

        # fill everything with hyphen
        latency_cs = ['-'] * len(entry_cs)

        # calculate total CS duration
        total_cs_duration = ['-']

        # calculate average latency
        avg_latency = ['-']

        # Calculate counts
        av_count = '-'
        esc_count = '-'
        fail_count = '-'

        # Calculate percentages
        av_perc = '-'
        esc_perc = '-'
        fail_perc = '-'

        entry_count_non_cs = '-'

        n_shuttl_per_ten_min_cs = '-'
        n_shuttl_per_ten_min_non_cs = '-'

        temp_df = pd.DataFrame(np.reshape([animal_id, av_count, esc_count, fail_count, av_perc, esc_perc, fail_perc, entry_count, entry_count_non_cs, n_shuttl_per_ten_min_cs, n_shuttl_per_ten_min_non_cs, total_duration] + entry_cs + exit_cs + latency_cs + total_cs_duration + avg_latency, (1, -1)), columns=col)

        return temp_df
    except Exception as e:
        file_info = ' '.join([file_path.split(os.sep)[-4].split()[-2], file_path.split(os.sep)[-3], file_path.split(os.sep)[-1].split(".")[-2]])
        if str(e) == 'index 0 is out of bounds for axis 0 with size 0' or str(e) == 'division by zero':
            print(f'[ERROR] Data processing failed for {file_info} :: Check if the file is empty or contains invalid data.')
        else:
            if len(entry_cs) == len(exit_cs) and len(entry_cs) == len(latency_cs) and len(exit_cs) == len(latency_cs) and (len(entry_cs) != 5 or len(entry_cs) != 11):
                print(f'[ERROR] Data processing failed for {file_info} :: Number of CS is {len(entry_cs)} which is invalid.')
            else:
                print(f'[ERROR] Data processing failed for {file_info} :: {e}')
        # print(traceback.format_exc())
        
        # sys.exit(1)
        return None

def process_gs_data(GS_DIR_PATH, time_discrepancy_dict):
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
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            ct, exp = file_path.split(os.sep)[-4].split()[-1], file_path.split(os.sep)[-3].upper()
            if ct not in time_discrepancy_dict or exp not in time_discrepancy_dict[ct]:
                return False
            temp_df = process_file(file_path, total_trials, col, time_discrepancy_dict)
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

def process_and_save_data(PATH, exp_df, ct, dt, time_discrepancy_dict, SAVE_DIR, add_animal_info=True): 
    """Process data in subfolders of PATH and save to Excel."""
    title_split = os.path.normpath(PATH).split(os.sep)  # Normalize the path to handle different OS path separators

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    info = title_split[-1].split()
    title = f"{info[1].split('_')[0]} {(info[1].split('_')[1]).upper()} {info[2]} {info[0]}"
    output_file = os.path.join(SAVE_DIR, f"{title}.xlsx")
    try:
        writer = ExcelWriter(output_file)
    except PermissionError:
        print(f"\nPermission denied to write to {title}.xlsx. This file may be open in another program. Please close the file and try again.")
        return

    for subfolder in sorted(os.listdir(PATH)):
        GS_DIR_PATH = os.path.join(PATH, subfolder)
        if os.path.isdir(GS_DIR_PATH) and ('SAA' in subfolder.upper() or 'LTM' in subfolder.upper()):
            GS_DIR_PATH = os.path.join(GS_DIR_PATH, 'csv files')
            data_df = process_gs_data(GS_DIR_PATH, time_discrepancy_dict)  # Assuming process_data is defined elsewhere
            if data_df is False:
                print(f'[ERROR] Data processing failed for {subfolder} :: Time discrepancy not found.')
                continue
            if add_animal_info:
                data_df = add_animal_details(data_df, exp_df, ct, dt)
                data_df.set_index('SN', inplace=True)
            data_df.sort_index(inplace=True)
            data_df.style.apply(align_center, axis=0).to_excel(writer, sheet_name=subfolder, index=True)
            print(f'[SUCCESS] Data processed for {subfolder}!')
        else:
            print(f'[SKIPPED] Not a valid SAA or LTM folder. {subfolder} skipped.')
        
    writer.close()

def extract_time_discrepancies(time_discrepancy_path):
    time_discrepancy_dict = defaultdict(lambda: defaultdict(lambda: 0))
    
    print('-'*50)
    print("Extracting time discrepancies...")
    print('-'*50)
    for file_name in os.listdir(time_discrepancy_path):
        if 'TimeDiscrepancy' in file_name:
            file_path = os.path.join(time_discrepancy_path, file_name)
            file_name = file_name.split('.')[0].split(' ')[-2]
            time_discrepancy_dict[file_name] = defaultdict(lambda: defaultdict(lambda: 0))
            
            with pd.ExcelFile(file_path) as xls:
                for sheet_name in xls.sheet_names:
                    sheet_df = pd.read_excel(xls, sheet_name)
                    sheet_df.iloc[:, 3] = sheet_df.iloc[:, 3].apply(lambda x: 0 if x < 3 else x)
                    
                    # Create the inner dictionary as a defaultdict
                    time_discrepancy_dict[file_name][sheet_name.upper()] = defaultdict(lambda: 0, 
                        {mouse_id: int(time_discrepancy) for mouse_id, time_discrepancy in zip(sheet_df.iloc[:, 0], sheet_df.iloc[:, 3]) if mouse_id is not None}
                    )

                    print(f"Time discrepancy extracted for {file_name} {sheet_name}!")

    print('-'*50)
    print("Time discrepancies extracted successfully!")
    print('-'*50)
    print()
    
    return time_discrepancy_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process data from subfolders in a folder and save to Excel.")
    parser.add_argument("--folder", help="Path to the data folder", type=str, required=True)
    parser.add_argument("--ct", help="Path to the Excel file containing experiment details", type=str, required=True)
    parser.add_argument("--time_discrepancy", help="Path to the folder containing time discrepancy Excel files", type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
    args = parser.parse_args()

    time_discrepancy_dict = extract_time_discrepancies(args.time_discrepancy)

    exp_df = pd.read_excel(args.ct, usecols=[0, 1, 2, 3, 4])
    exp_df.columns = ['SN', 'Animal', 'Sex', 'Subject ID', 'Group ']

    for subfolder in sorted(os.listdir(args.folder)):
        ct = subfolder.split()[-1] # if using old WT SAA data, use subfolder.split()[-1]. For newer data following established naming convention, use subfolder.split()[-2]
        dt = subfolder.split()[0]
        GS_DIR_PATH = os.path.join(args.folder, subfolder)
        try:
            if os.path.isdir(GS_DIR_PATH):
                print(f'[CURRENT] Processing data for {os.path.split(GS_DIR_PATH)[-1]}...')
                print('-'*50)
                process_and_save_data(GS_DIR_PATH, exp_df, ct, dt, time_discrepancy_dict, args.output, add_animal_info=True)
                
        except Exception as e:
            # print(traceback.format_exc()) # Uncomment this line to print traceback to help debug errors
            # print(f"Error processing {os.path.split(GS_DIR_PATH)[-1]} ::\n\n {e}\n")
            print(f'[ERROR] Data processing failed for {os.path.split(GS_DIR_PATH)[-1]} :: {e}')
        print('-'*50)