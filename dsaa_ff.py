import pandas as pd
import numpy as np
import os
from pandas import ExcelWriter
import warnings
import traceback
import sys
import argparse

warnings.filterwarnings("ignore")

class FreezeFrame:
    def __init__(self, timestamps_path, ct_path, folder_path, output_path):
        '''Function to initialize the class with the paths to the timestamps file, CT file, folder containing the FreezeFrame data, and the output folder.'''
        self.timestamps_path = timestamps_path
        self.ct_path = ct_path
        self.folder_path = folder_path
        self.output_path = output_path
        self.output = self.output_path
        self.timestamps = self.process_sheets()
        self.counter = 0

    def get_cols(self, num_of_cs):
        return pd.MultiIndex.from_arrays([['Animal ID', ' '] + ['CS+']*(num_of_cs) + ['CS-']*(num_of_cs) + ['D.I.'], ['', 'Threshold'] + [str(i) for i in range(1, num_of_cs)] + ['Mean CS+'] + [str(i) for i in range(1, num_of_cs)] + ['Mean CS-'] + [' ']])
    
    def get_cohort_data(self, ct):
        '''Function to extract the cohort data from the CT file.'''
        df = pd.read_excel(self.ct_path, usecols=range(5))
        ct_row_index = df.index[df['Unnamed: 0'].str.contains(ct, na=False)].tolist()[0]

        # Extract rows following the CT row until a row with all NaN values is encountered
        new_df_rows = []
        for i in range(ct_row_index+1, len(df)):
            if df.iloc[i].isnull().all():
                break
            new_df_rows.append(df.iloc[i])

        # Create a new DataFrame with the extracted rows
        new_df = pd.DataFrame(new_df_rows[1:])
        new_df.columns = new_df_rows[0].tolist()

        # drop first col
        new_df.drop(new_df.columns[0], axis=1, inplace=True)

        new_df.reset_index(drop=True, inplace=True) # reset index
        new_df.columns = pd.MultiIndex.from_arrays([new_df.columns, [''] + ['']*len(new_df.columns[1:])]) # set multi-level columns
        new_df_renamed = new_df.rename(columns={'Animal': 'Animal ID'}) # rename columns
        return new_df_renamed

    def align_center(self, x):
        '''Function to align the text in the cells to the center.'''
        return ['text-align: center' for _ in x]

    def process_folder(self):
        '''Function to process the folder containing the FreezeFrame data.'''
        subfolders = [f for f in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, f))] # get all subfolders
        for subfolder in subfolders: # for each subfolder
            if 'archive' in subfolder.lower() or 'timestamps' in subfolder.lower():
                continue
            ct = subfolder.split()[-2] # extract the CT from the subfolder name
            print('\n=============================', ct, '=============================')
            self.ct_df = self.get_cohort_data(ct) # get the cohort data for the CT
            try:
                self.process_subfolder(subfolder) # process the FreezeFrame data for the subfolder
            except Exception as e: # if there is an exception, print the exception
                if str(e) == "At least one sheet must be visible":
                    print('No proper sheets found in the file. Skipping... Please check the file.')
                else:
                    # print(traceback.format_exc())
                    print(f'{ct} -> {e}')

    def process_subfolder(self, subfolder):
        '''Function to process the FreezeFrame data for each subfolder.'''
        output_path = os.path.join(self.output, subfolder + '.xlsx') # set the output path
        
        ct = subfolder.split()[-2] # extract the CT from the subfolder name
        writer = ExcelWriter(output_path) # create an ExcelWriter object
        for file in os.listdir(os.path.join(self.folder_path, subfolder)): # for each file in the subfolder
            if file.endswith('.csv'): # if the file is a CSV file
                sheet_name = file.split('\\')[-1].split('.')[-2].split('_')[-1] # extract the sheet name
                if 'SAA' not in sheet_name and 'LTM' not in sheet_name: # if the sheet name does not contain 'CT' or 'LTM'
                    print('\nSkipping:', sheet_name)
                    continue

                print('\nProcessing: ', sheet_name)
                file_path = os.path.join(self.folder_path, subfolder, file) # set the file path
                try:
                    data = self.process_file(file_path, sheet_name, ct) # process the FreezeFrame data
                    final = pd.merge(self.ct_df, data, on='Animal ID', how='inner') # merge the cohort data with the FreezeFrame data
                    final.index += 1
                    final.style.apply(self.align_center, axis=0).to_excel(writer, sheet_name=sheet_name, index=True) # write the data to the Excel file
                    print('File #', self.counter, ' processed: ', file)
                except Exception as e: # if there is an exception, print the exception
                    if type(e) == KeyError:
                        print(f'{e} not found in the timestamps file for {ct}. Skipping...')
                        print(f'File # {self.counter} NOT processed: {file}')
                        continue
                    else:
                        print(f'{ct} -> {e}')
                        # print(traceback.format_exc())
                        print(f'File # {self.counter} NOT processed: {file}')
                        continue
                finally:
                    self.counter += 1
        writer.close() # close the ExcelWriter object

    def clean_columns(self, columns):
        '''Function to clean the column names.'''
        # Remove any leading or trailing whitespaces
        columns = [col.strip() for col in columns]
        # convert all columns to integers if possible
        columns = [int(float(column)) if column.replace('.', '').replace('-', '').replace('+', '').replace('e', '').isdigit() else column for column in columns]
        return columns

    def process_file(self, file_path, experiment_name, ct):
        '''Function to process the FreezeFrame data for each file.'''
        ff_df = pd.read_csv(file_path, header=1) # read the CSV file
        ff_df.columns = self.clean_columns(list(ff_df.columns)) # clean the column names
        return self.process_experiment(ff_df, experiment_name, ct)
    
    def process_experiment(self, ff_df, experiment_name, ct):
        '''Function to process the FreezeFrame data for the given timestamps.'''
        experiment_name = experiment_name.upper()
        len_of_cols = 0
        try:
            len_of_cols = max([len(self.timestamps[ct][experiment_name][key]['onset']['cs_plus']) for key in self.timestamps[ct][experiment_name].keys()])
        except IndexError:
            print('No data found for CT: ', ct, ' and experiment: ', experiment_name)
            return pd.DataFrame(columns=self.get_cols(6 if 'LTM' in experiment_name else 11))
            
        df = pd.DataFrame(columns=self.get_cols(len_of_cols+1))
    
        for animal_id in ff_df.iloc[1:, 0]: # for each animal ID
            if str(animal_id) == 'nan': # if the animal ID is 'nan', skip it
                continue
            threshold = ff_df[ff_df.iloc[:, 0].astype(str).str.contains(str(animal_id))].loc[:, 'Threshold'].values[0] # extract the threshold
            try:
                cs_plus_start = self.timestamps[ct][experiment_name][animal_id]['onset']['cs_plus'] # extract the CS start timestamps
            except KeyError:
                print('Animal ID: ', animal_id, ' in CT: ', ct, ' not found in the timestamps file. Skipping...')
                continue
            cs_plus_end = self.timestamps[ct][experiment_name][animal_id]['offset']['cs_plus'] # extract the CS end timestamps
            cs_minus_start = self.timestamps[ct][experiment_name][animal_id]['onset']['cs_minus'] # extract the CS start timestamps
            cs_minus_end = self.timestamps[ct][experiment_name][animal_id]['offset']['cs_minus'] # extract the CS end timestamps
            cs_plus = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(cs_plus_start, cs_plus_end)] # get the average of the CS+ data
            cs_minus = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(cs_minus_start, cs_minus_end)] # get the average of the CS- data
            mean_cs_plus = round(np.mean([c for c in cs_plus if c != 999 and c != 'NA']), 2) # calculate the mean of the CS+ data
            mean_cs_minus = round(np.mean([c for c in cs_minus if c != 999 and c != 'NA']), 2) # calculate the mean of the CS- data

            di = self.calculate_di(mean_cs_plus, mean_cs_minus) # calculate the D.I.

            if len(cs_plus_start) < len_of_cols:
                cs_plus.append('NA')
                cs_minus.append('NA')

            data = [animal_id.split()[-1], threshold, *cs_plus, mean_cs_plus, *cs_minus, mean_cs_minus, di] # create the data
            
            df = pd.concat([df, pd.DataFrame([data], columns=self.get_cols(len_of_cols+1))], ignore_index=True) # concatenate the data to the DataFrame
        return df # return the DataFrame

    def calculate_di(self, mean_cs_plus, mean_cs_minus):
        '''Function to calculate the D.I.'''
        try: # try to calculate the D.I.
            return round((mean_cs_plus - mean_cs_minus) / (mean_cs_plus + mean_cs_minus), 2) # return the D.I.
        except ZeroDivisionError: # if there is a ZeroDivisionError, return 'N/A'
            return 'N/A'

    def get_ff_avg(self, animal_id, start, end, ff_df):
        '''Function to calculate the average of the FreezeFrame data for the given animal ID for the given start and end timestamps.'''
        try:
            first_column_str = ff_df.iloc[:, 0].astype(str)
            matches_exact_animal_id = first_column_str == animal_id
            sub_df = ff_df.loc[matches_exact_animal_id, int(start):int(end)]
            sub_df = sub_df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x).replace("NaN", 0).apply(pd.to_numeric, errors='coerce') # clean the data by first stripping the strings, replacing "NaN" with 0, and converting the data to numeric
            return float(sub_df.mean(axis=1).round(2)) # return the average of the data
        except Exception as e: # if there is an exception, return 999
            if str(e) == "cannot convert the series to <class 'float'>":
                print('Multiple entries for animal ID: ', animal_id)
                return 999
            else:
                print('Animal ID: ', animal_id, ' -> ', e)
                return 'NA'
        
    def parse_sheet(self, xlsx, sheet):
        '''Function to parse the sheet and extract the data.'''
        df = xlsx.parse(sheet)
        vals = {}
        try:
            animal_ids = list(df.iloc[2:, 1].values)
            cs_plus_onsets = list(df.loc[2:, 'entry':'exit'].iloc[:, :-1].values)
            # check for nan. if they are present, then remove them
            cs_plus_onsets = [[i for i in sublist if not pd.isnull(i)] for sublist in cs_plus_onsets]
            cs_plus_offsets = list(df.loc[2:, 'exit':'latency'].iloc[:, :-1].values)
            cs_plus_offsets = [[i for i in sublist if not pd.isnull(i)] for sublist in cs_plus_offsets]
            cs_minus_onsets = list(df.loc[2:, 'entry.1':'exit.1'].iloc[:, :-1].values)
            cs_minus_onsets = [[i for i in sublist if not pd.isnull(i)] for sublist in cs_minus_onsets]
            cs_minus_offsets = list(df.loc[2:, 'exit.1':'latency.1'].iloc[:, :-1].values)
            cs_minus_offsets = [[i for i in sublist if not pd.isnull(i)] for sublist in cs_minus_offsets]
            for animal_id, cs_plus_onset, cs_plus_offset, cs_minus_onset, cs_minus_offset in zip(animal_ids, cs_plus_onsets, cs_plus_offsets, cs_minus_onsets, cs_minus_offsets):
                vals[animal_id] = {'onset': {'cs_plus': list(cs_plus_onset), 'cs_minus': list(cs_minus_onset)}, 'offset': {'cs_plus': list(cs_plus_offset), 'cs_minus': list(cs_minus_offset)}}
        except KeyError:
            print('No data in the sheet: ', sheet)
            return None
        return vals
    
    def process_sheets(self):
        all_files = os.listdir(self.timestamps_path)
        all_data = {}
        for file in all_files:
            if file.endswith('.xlsx'):
                ct = file.split()[-2]
                if 'CT' not in ct:
                    continue
                xlsx = pd.ExcelFile(os.path.join(self.timestamps_path, file))
                dfs = {sheet.upper() : self.parse_sheet(xlsx, sheet) for sheet in xlsx.sheet_names}
                all_data[ct] = dfs

        return all_data

def main():
    '''Function to parse the command line arguments and process the FreezeFrame data.'''
    parser = argparse.ArgumentParser(description='Process FreezeFrame data')
    parser.add_argument('--timestamps', type=str, required=True, help='Path to the timestamps file')
    parser.add_argument('--ct', type=str, required=True, help='Path to the CT file')
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing the FreezeFrame data')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
    args = parser.parse_args()
    timestamps_path = args.timestamps
    ct_path = args.ct
    folder_path = args.folder
    output_path = args.output

    ff = FreezeFrame(timestamps_path, ct_path, folder_path, output_path) # create a FreezeFrame object
    ff.process_folder() # process the folder containing the FreezeFrame data

# Run the main function
if __name__ == '__main__':
    main()