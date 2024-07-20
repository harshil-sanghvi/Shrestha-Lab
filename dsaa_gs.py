import os
import pandas as pd
import numpy as np
from pandas import ExcelWriter
import warnings
import traceback
import argparse

warnings.filterwarnings("ignore")

def check_saa_in_path(path):
    """Check if 'SAA' is present in the path."""
    return 'SAA' in path.split(os.path.sep)[-2].upper()

def get_num_trials(path):
    if not check_saa_in_path(path):
        return 5
    else:
        first_file = os.listdir(path)[0]
        first_file_path = os.path.join(path, first_file)
        return get_trials_from_file(first_file_path)
    
def get_trials_from_file(file_path):
    df = pd.read_csv(file_path, header=None).iloc[:, 1:5]
    df.columns = ['Time', 'Event', 'C', 'Desc']
    return df[(df['Desc'] == 'CS+ (Threat cue)')].shape[0]//2


def get_total_duration(df):
    return df[df['Event'] == 'Finish']['Time'].astype(int).values[0]

def get_n_shuttl_total(df):
    return df[(df['Desc'] == 'Right Entry') | (df['Desc'] == 'Left Entry')].shape[0]

def get_entry_exit_cs(df):
    entry_cs = df[df['Event'] == 'Entry']['Seconds'].tolist()
    exit_cs = df[df['Event'] == 'Exit']['Seconds'].tolist()
    return entry_cs, exit_cs

def get_latency(entry_cs, exit_cs):
    return [exit - entry for exit, entry in zip(exit_cs, entry_cs)]

def get_total_cs_duration(latency_cs):
    return [sum(latency_cs)]

def get_avg_latency(latency_cs):
    return [round(sum(latency_cs) / len(latency_cs), 1)]

def get_n_shuttl_per_ten_min(n_shuttl, total_duration):
    return round((n_shuttl / total_duration) * 600, 2)

def get_num_of_av_esc_fail(df, total_trials, avoid, escape):
    av = df[df['Desc'] == avoid].shape[0] // 2
    esc = df[df['Desc'] == escape].shape[0] // 2
    fail = total_trials - av - esc
    return av, esc, fail

def get_perc_of_av_esc_fail(av, esc, total_trials):
    av_perc = round(av / total_trials * 100, 1)
    esc_perc = round(esc / total_trials * 100, 1)
    fail_perc = abs(round(100 - av_perc - esc_perc, 1))
    return av_perc, esc_perc, fail_perc

def get_animal_id(file_path):
    return os.path.basename(file_path).split('_')[-1].split('.')[0].split()[-1]

def get_cs_data(cs_df):
    entry_cs, exit_cs = get_entry_exit_cs(cs_df)
    latency_cs = get_latency(entry_cs, exit_cs)
    total_cs_duration = get_total_cs_duration(latency_cs)
    avg_latency = get_avg_latency(latency_cs)
    return entry_cs, exit_cs, latency_cs, total_cs_duration, avg_latency

def get_duration(latencies):
    return sum(latencies)

def calculate_DI(cs_pos_av, cs_neg_av):
    try:
        return [round((cs_pos_av - cs_neg_av) / (cs_pos_av + cs_neg_av), 1)]
    except ZeroDivisionError:
        return ['N/A']

def get_features_df(animal_id, cs_pos_av, cs_pos_esc, cs_pos_fail, cs_pos_av_perc, cs_pos_esc_perc, cs_pos_fail_perc, n_shuttl_total, n_shuttl_non_cs, n_shuttl_per_ten_min_cs_pos, n_shuttl_per_ten_min_non_cs, normalized_fraction_shuttling_cs_pos, normalized_fraction_shuttling_cs_neg, normalized_fraction_shuttling_non_cs, total_duration, cs_pos_entry, cs_pos_exit, cs_pos_latency, cs_pos_total_duration, cs_pos_avg_latency, cs_neg_av, cs_neg_esc, cs_neg_fail, cs_neg_av_perc, cs_neg_esc_perc, cs_neg_fail_perc, n_shuttl_per_ten_min_cs_neg, cs_neg_entry, cs_neg_exit, cs_neg_latency, cs_neg_total_duration, cs_neg_avg_latency, di, col):
    return pd.DataFrame(np.reshape([animal_id, cs_pos_av, cs_pos_esc, cs_pos_fail, cs_pos_av_perc, cs_pos_esc_perc, cs_pos_fail_perc] + cs_pos_entry + cs_pos_exit + cs_pos_latency + cs_pos_total_duration + cs_pos_avg_latency + [cs_neg_av, cs_neg_esc, cs_neg_fail, cs_neg_av_perc, cs_neg_esc_perc, cs_neg_fail_perc, n_shuttl_total, n_shuttl_non_cs, n_shuttl_per_ten_min_cs_pos, n_shuttl_per_ten_min_cs_neg, n_shuttl_per_ten_min_non_cs, normalized_fraction_shuttling_cs_pos, normalized_fraction_shuttling_cs_neg, normalized_fraction_shuttling_non_cs, total_duration] + cs_neg_entry + cs_neg_exit + cs_neg_latency + cs_neg_total_duration + cs_neg_avg_latency + di, (1, -1)), columns=col)
    
def process_file(file_path, total_trials, col):
    """Process an individual file and extract required data."""
    try:
        df = pd.read_csv(file_path, header=None).iloc[:, 1:5]
        df.columns = ['Time', 'Event', 'C', 'Desc']

        if total_trials == 12:
            df = df.iloc[:df[(df['Desc'] == 'CS- (Safety cue)') & (df['Event'] == 'Exit')].index[10] + 1]
            total_duration = df[(df['Event'] == 'Exit') & (df['Desc'] == 'CS- (Safety cue)')]['Time'].iloc[10].astype(int)
        else:
            total_duration = get_total_duration(df)
        
        n_shuttl_total = get_n_shuttl_total(df)

        cs_pos_df = df[(df['Desc'] == 'CS+ (Threat cue)')]
        cs_pos_df['Seconds'] = cs_pos_df['Time'].astype(int)

        cs_pos_entry, cs_pos_exit, cs_pos_latency, cs_pos_total_duration, cs_pos_avg_latency = get_cs_data(cs_pos_df)

        cs_neg_df = df[(df['Desc'] == 'CS- (Safety cue)')]
        cs_neg_df['Seconds'] = cs_neg_df['Time'].astype(int)

        cs_neg_entry, cs_neg_exit, cs_neg_latency, cs_neg_total_duration, cs_neg_avg_latency = get_cs_data(cs_neg_df)

        trials_for_this_file = min(get_trials_from_file(file_path), 11)
        
        cs_pos_av, cs_pos_esc, cs_pos_fail = get_num_of_av_esc_fail(df, trials_for_this_file, 'CS+ Avoidance', 'CS+ Escape')
        cs_neg_av, cs_neg_esc, cs_neg_fail = get_num_of_av_esc_fail(df, trials_for_this_file, 'CS- Avoidance', 'CS- Escape')
        
        cs_pos_av_perc, cs_pos_esc_perc, cs_pos_fail_perc = get_perc_of_av_esc_fail(cs_pos_av, cs_pos_esc, trials_for_this_file)
        cs_neg_av_perc, cs_neg_esc_perc, cs_neg_fail_perc = get_perc_of_av_esc_fail(cs_neg_av, cs_neg_esc, trials_for_this_file)
        
        n_shuttl_non_cs = n_shuttl_total - cs_pos_av - cs_neg_av
        
        cs_pos_duration = get_duration(cs_pos_latency)
        cs_neg_duration = get_duration(cs_neg_latency)

        n_shuttl_per_ten_min_cs_pos = get_n_shuttl_per_ten_min(cs_pos_av, cs_pos_duration)
        n_shuttl_per_ten_min_cs_neg = get_n_shuttl_per_ten_min(cs_neg_av, cs_neg_duration)
        n_shuttl_per_ten_min_non_cs = get_n_shuttl_per_ten_min(n_shuttl_non_cs, total_duration - cs_pos_duration - cs_neg_duration)

        normalized_fraction_shuttling_cs_pos = n_shuttl_per_ten_min_cs_pos / (n_shuttl_per_ten_min_non_cs + n_shuttl_per_ten_min_cs_pos + n_shuttl_per_ten_min_cs_neg)
        normalized_fraction_shuttling_cs_neg = n_shuttl_per_ten_min_cs_neg / (n_shuttl_per_ten_min_non_cs + n_shuttl_per_ten_min_cs_pos + n_shuttl_per_ten_min_cs_neg)
        normalized_fraction_shuttling_non_cs = n_shuttl_per_ten_min_non_cs / (n_shuttl_per_ten_min_non_cs + n_shuttl_per_ten_min_cs_pos + n_shuttl_per_ten_min_cs_neg)
           
        # round to two decimal places
        normalized_fraction_shuttling_non_cs = round(normalized_fraction_shuttling_non_cs, 2)
        normalized_fraction_shuttling_cs_pos = round(normalized_fraction_shuttling_cs_pos, 2)
        normalized_fraction_shuttling_cs_neg = round(normalized_fraction_shuttling_cs_neg, 2)

        DI = calculate_DI(cs_pos_av, cs_neg_av)

        animal_id = get_animal_id(file_path)

        if trials_for_this_file == 10:
            cs_pos_entry.append('N/A')
            cs_pos_exit.append('N/A')
            cs_pos_latency.append('N/A')
            cs_neg_entry.append('N/A')
            cs_neg_exit.append('N/A')
            cs_neg_latency.append('N/A')

        return get_features_df(animal_id, cs_pos_av, cs_pos_esc, cs_pos_fail, cs_pos_av_perc, cs_pos_esc_perc, cs_pos_fail_perc, n_shuttl_total, n_shuttl_non_cs, n_shuttl_per_ten_min_cs_pos, n_shuttl_per_ten_min_non_cs, normalized_fraction_shuttling_cs_pos, normalized_fraction_shuttling_cs_neg, normalized_fraction_shuttling_non_cs, total_duration, cs_pos_entry, cs_pos_exit, cs_pos_latency, cs_pos_total_duration, cs_pos_avg_latency, cs_neg_av, cs_neg_esc, cs_neg_fail, cs_neg_av_perc, cs_neg_esc_perc, cs_neg_fail_perc, n_shuttl_per_ten_min_cs_neg, cs_neg_entry, cs_neg_exit, cs_neg_latency, cs_neg_total_duration, cs_neg_avg_latency, DI, col)
    except Exception as e:
        print(file_path.split(os.path.sep)[-4].split()[-2], file_path.split(os.path.sep)[-3], file_path.split(os.path.sep)[-1].split('.')[-2]," -> ", e)
        return None

def process_gs_data(GS_DIR_PATH):
    """Process data from files in GS_DIR_PATH."""
    # Check if GS_DIR_PATH contains 'SAA'
    total_trials = get_num_trials(GS_DIR_PATH)

    trials_for_col = 5 if total_trials == 5 else 11

    # Define column names for the DataFrame
    col = pd.MultiIndex.from_arrays([
        ['Animal ID'] + ['CS+ #']*3 + ['CS+ %']*3 + ['entry']*trials_for_col + ['exit']*trials_for_col + ['latency']*trials_for_col + ['total CS+'] + ['Avg CS+'] + ['CS- #']*3 + ['CS- %']*3 + ['n[Shuttle]']*2 + ['n[Shuttle]/10m']*3 + ['Normalized Fraction Shuttling']*3 + ['Total'] + ['entry']*trials_for_col + ['exit']*trials_for_col + ['latency']*trials_for_col + ['total CS-'] + ['Avg CS-'] + ['DI'], 
        [''] + ['Av', 'Esc', 'Fail']*2 + ['CS+ ' + str(i) for i in range(1, trials_for_col + 1)]*3 + ['Duration', 'Latency'] + ['Av', 'Esc', 'Fail']*2 + ['Total', 'Non-CS', 'CS+', 'CS-', 'Non-CS', 'CS+', 'CS-', 'Non-CS'] + ['Duration'] + ['CS- ' + str(i) for i in range(1, trials_for_col + 1)]*3 + ['Duration', 'Latency', '']
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
    try:
        start_index = exp_df[~na_mask & exp_df['SN'].str.contains(ct) & exp_df['SN'].str.contains(dt)].index[0] # Get the index of the first non-NaN value in SN column
    except IndexError:
        raise Exception(f"Could not find the start index for {ct} {dt}. Please check the experiment details file and look specifically for the row before the actual data starts. If the issue persists, try debugging by comparing the format of experiment details file with the format adhered to in the script.")

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
    title_split = PATH.split(os.path.sep)
    SAVE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    info = title_split[-1].split()
    title = info[1].split('_')[0] + ' ' + (info[1].split('_')[1]).upper() + ' ' + info[2] + ' ' + info[0]
    output_file = os.path.join(SAVE_DIR, f"{title}.xlsx")
    try:
        writer = ExcelWriter(output_file)
    except PermissionError:
        print(f"\nPermission denied to write to {title}.xlsx. This file may be open in another program. Please close the file and try again.")
        return

    for subfolder in sorted(os.listdir(PATH)):
        wasSuccessful = True
        try:
            GS_DIR_PATH = os.path.join(PATH, subfolder)
            if os.path.isdir(GS_DIR_PATH) and ('SAA' in subfolder.upper() or 'LTM' in subfolder.upper()):
                GS_DIR_PATH = os.path.join(GS_DIR_PATH, 'csv files')
                data_df = process_gs_data(GS_DIR_PATH)  # Assuming process_data is defined elsewhere
                if add_animal_info:
                    data_df = add_animal_details(data_df, exp_df, ct, dt)
                    data_df.set_index('SN', inplace=True)
                    data_df.style.apply(align_center, axis=0).to_excel(writer, sheet_name=subfolder, index=True)
            else:
                wasSuccessful = False
                print(f"Skipping {subfolder} as it is not a valid subfolder.")
        except Exception as e:
            # if type of error is keyerror then print that most likely it is that there are multiple CSV files for same animal ID in experiment subfolder
            if isinstance(e, KeyError):
                print(f"Most likely there are multiple CSV files for the same animal ID in the experiment {subfolder}. Please check the data and try again. If the issue persists, try debugging by printing columns and index of the data_df.")
            elif 'csv files' in str(e) and 'No such file or directory' in str(e):
                print(f'There is no folder named "csv files" in the experiment {subfolder}. Please check the folder structure and try again.')
            else:
                print(f"Error processing {subfolder}: {e}")
            wasSuccessful = False
        if wasSuccessful:
            print(f'\t{subfolder} processed successfully!')

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process data from subfolders in a folder and save to Excel.")
    parser.add_argument("--path", help="Path to the data folder", type=str, required=True)
    parser.add_argument("--exp_details_path", help="Path to the Excel file containing experiment details", type=str, required=True)
    args = parser.parse_args()

    exp_df = pd.read_excel(args.exp_details_path, usecols=[0, 1, 2, 3, 4])
    exp_df.columns = ['SN', 'Animal', 'Sex', 'Subject ID', 'Group ']

    for subfolder in sorted(os.listdir(args.path)):
        ct = subfolder.split()[-2] # if using old WT SAA data, use subfolder.split()[-1]. For newer data following established naming convention, use subfolder.split()[-2]
        dt = subfolder.split()[0]
        GS_DIR_PATH = os.path.join(args.path, subfolder)
        try:
            if os.path.isdir(GS_DIR_PATH):
                process_and_save_data(GS_DIR_PATH, exp_df, ct, dt, add_animal_info=True)
                print(f"Data processed successfully for {subfolder}!\n")
        except Exception as e:
            print(f"Error processing {GS_DIR_PATH.split(os.path.sep)[-1]} ::\n\n {e}\n")