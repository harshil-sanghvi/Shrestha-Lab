import pandas as pd
import numpy as np
import os
from pandas import ExcelWriter
import warnings
import sys
import re
import argparse

warnings.filterwarnings("ignore")

class FreezeFrame:
    def __init__(self, timestamps_path, ct_path, folder_path, output_path):
        '''Function to initialize the class with the paths to the timestamps file, CT file, folder containing the FreezeFrame data, and the output folder.'''
        self.timestamps_path = timestamps_path
        self.ct_path = ct_path
        self.folder_path = folder_path
        self.output_path = output_path
        self.training_timestamps, self.ltm_timestamps = None, None
        self.output = self.output_path
        self.experiment_name = None
        
    def get_cols(self, experiment):
        '''Function to get the column names for the given experiment.'''
        if experiment == 7:
            return pd.MultiIndex.from_arrays([['Animal ID', ' ', ' '] + ['CS+']*4 + ['ITI']*3 + [' '],
                                          ['', 'Threshold', 'Pre-CS'] + [str(i) for i in range(1, 4)] + ['Mean CS+'] + [str(i) for i in range(1, 3)] + ['Mean ITI', 'Post-CS']])
        
        return pd.MultiIndex.from_arrays([['Animal ID', ' ', ' '] + ['CS+']*3 + ['ITI']*2 + [' '],
                                          ['', 'Threshold', 'Pre-CS'] + [str(i) for i in range(1, 3)] + ['Mean CS+'] + [str(i) for i in range(1, 2)] + ['Mean ITI', 'Post-CS']])
        

    def process_timestamps(self):
        '''Function to process the timestamps file and extract the training and LTM timestamps.'''
        df = pd.read_excel(self.timestamps_path)

        # Find the index where "Epoch" occurs to split the DataFrame
        split_index = df.index[df["Unnamed: 0"] == "Epoch"].tolist()

        # Split the DataFrame into two based on the split index
        training_df = df.iloc[2:split_index[1]-1, :].reset_index(drop=True)
        ltm_df = df.iloc[split_index[1]:, :].reset_index(drop=True)

        # Drop the unnecessary rows with NaN values in the first column
        training_df = training_df.dropna(subset=['Unnamed: 0'], how='all')
        ltm_df = ltm_df.dropna(subset=['Unnamed: 0'], how='all')

        # Set column names
        training_df.columns = training_df.iloc[0]
        ltm_df.columns = ltm_df.iloc[0]

        # Drop the first row as it's just a repetition of column names
        training_df = training_df.iloc[1:].reset_index(drop=True)
        ltm_df = ltm_df.iloc[1:].reset_index(drop=True)

        self.training_timestamps = training_df
        self.ltm_timestamps = ltm_df
    
    def get_cohort_data(self, ct):
        '''Function to extract the cohort data from the CT file.'''
        df = pd.read_excel(self.ct_path, usecols=range(5))
        
        ct_row_index = df.index[df.iloc[:, 0].str.contains(ct, na=False)].tolist()[0]

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
            ct = subfolder.split()[-2] # extract the CT from the subfolder name
            self.ct_df = self.get_cohort_data(ct) # get the cohort data for the CT
            self.process_subfolder(subfolder) # process the FreezeFrame data for the subfolder
            print(subfolder, 'processed successfully!')

    def process_subfolder(self, subfolder):
        '''Function to process the FreezeFrame data for each subfolder.'''
        output_path = os.path.join(self.output, subfolder + '.xlsx') # set the output path
        writer = ExcelWriter(output_path) # create an ExcelWriter object
        for file in os.listdir(os.path.join(self.folder_path, subfolder)): # for each file in the subfolder
            if not file.endswith('.csv'):
                continue
            sheet_name = file.split(os.sep)[-1].split('.')[-2] # extract the sheet name
            if file.endswith('.csv'): # if the file is a CSV file
                file_path = os.path.join(self.folder_path, subfolder, file) # set the file path
                data = self.process_file(file_path, sheet_name) # process the FreezeFrame data
                final = pd.merge(self.ct_df, data, on='Animal ID', how='inner') # merge the cohort data with the FreezeFrame data
                # if merged data is empty then save the data as it is
                if final.empty:
                    data.style.apply(self.align_center, axis=0).to_excel(writer, sheet_name=sheet_name.split('_')[-1], index=True)
                    continue
                final.style.apply(self.align_center, axis=0).to_excel(writer, sheet_name=sheet_name.split('_')[-1], index=True) # write the data to the Excel file
        writer.close() # close the ExcelWriter object

    def clean_columns(self, columns):
        '''Function to clean the column names.'''
        # Remove any leading or trailing whitespaces
        columns = [col.strip() for col in columns]
        # convert all columns to integers if possible
        columns = [int(float(column)) if column.replace('.', '').replace('-', '').replace('+', '').replace('e', '').isdigit() else column for column in columns]
        return columns

    def process_file(self, file_path, experiment_name):
        '''Function to process the FreezeFrame data for each file.'''
        ff_df = pd.read_csv(file_path, header=1) # read the CSV file
        ff_df.columns = self.clean_columns(list(ff_df.columns)) # clean the column names
        self.experiment_name = experiment_name.split('_')[-1].lower() # extract the experiment name
        if 'ltm' in self.experiment_name: # if the experiment is LTM
            return self.process_ltm(ff_df) # process the FreezeFrame data for LTM
        return self.process_training(ff_df) # process the FreezeFrame data for training
    
    def process_training(self, ff_df):
        '''Function to process the FreezeFrame data for the training experiment.'''
        return self.process_data(ff_df, self.training_timestamps)
      
    def process_ltm(self, ff_df):
        '''Function to process the FreezeFrame data for the LTM experiment.'''
        return self.process_data(ff_df, self.ltm_timestamps)
    
    def process_data(self, ff_df, timestamps):
        '''Function to process the FreezeFrame data for the given timestamps.'''
        df = pd.DataFrame(columns=self.get_cols(len(timestamps))) # create a DataFrame with the column names
        pre_cs_start, pre_cs_end = self.extract_timestamps(timestamps, 'Pre-CS') # extract the start and end timestamps for Pre-CS
        cs_plus_start, cs_plus_end = self.extract_timestamps(timestamps, r'CS\+') # extract the start and end timestamps for CS+
        iti_start, iti_end = self.extract_timestamps(timestamps, 'ITI') # extract the start and end timestamps for ITI
        post_cs_start, post_cs_end = self.extract_timestamps(timestamps, 'Post-CS') # extract the start and end timestamps for Post-CS

        for animal_id in ff_df.iloc[1:, 0]: # for each animal ID
            if str(animal_id) == 'nan': # if the animal ID is 'nan', skip it
                continue
            
            pattern = f"^{re.escape(str(animal_id))}$"
            threshold = ff_df[ff_df.iloc[:, 0].astype(str).str.contains(pattern)].loc[:, 'Threshold'].values[0] # extract the threshold

            pre_cs = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(pre_cs_start, pre_cs_end)] # extract the average of the FreezeFrame data for Pre-CS
            cs_plus = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(cs_plus_start, cs_plus_end)] # extract the average of the FreezeFrame data for CS+
            iti = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(iti_start, iti_end)] # extract the average of the FreezeFrame data for ITI
            post_cs = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(post_cs_start, post_cs_end)] # extract the average of the FreezeFrame data for Post-CS
            mean_cs_plus = round(np.mean(cs_plus), 2) # calculate the mean of the CS+ data
            mean_iti = round(np.mean(iti), 2) # calculate the mean of the ITI data

            data = [animal_id.split()[-1], threshold, pre_cs[0], *cs_plus, mean_cs_plus, *iti, mean_iti, *post_cs] # create the data list            
            df = pd.concat([df, pd.DataFrame([data], columns=self.get_cols(len(timestamps)))], ignore_index=True) # concatenate the data to the DataFrame
        return df # return the DataFrame
        
    def extract_timestamps(self, timestamps, label):
        '''Function to extract the start and end timestamps for the given label.'''
        start = timestamps[timestamps['Epoch'].str.contains(label)]['Onset'].values # extract the start timestamps
        end = timestamps[timestamps['Epoch'].str.contains(label)]['Offset'].values # extract the end timestamps
        return start, end # return the start and end timestamps

    def get_ff_avg(self, animal_id, start, end, ff_df):
        '''Function to calculate the average of the FreezeFrame data for the given animal ID for the given start and end timestamps.'''
        try:
            pattern = f"^{re.escape(str(animal_id))}$"
            sub_df = ff_df.loc[ff_df.iloc[:, 0].astype(str).str.contains(pattern), start:end] # extract the FreezeFrame data for the given animal ID and timestamps
            sub_df = sub_df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x).replace("NaN", 0).apply(pd.to_numeric, errors='coerce') # clean the data by first stripping the strings, replacing "NaN" with 0, and converting the data to numeric
            return float(sub_df.mean(axis=1).round(2)) # return the average of the data
        except Exception as e: # if there is an exception, return 0
            if e.args[0].startswith("cannot convert the series to "):
                print(f'Multiple values for {animal_id}')
            elif e.args[0].startswith("cannot do slice indexing"):
                timestamp = int(e.args[0].split()[-4][1:-1])
                print(f'No values for {animal_id} for timestamp value {timestamp} in {self.experiment_name}')
            else:
                print(f'Error for {animal_id} in {start} to {end} -> {e}')
            return 0
        
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
    ff.process_timestamps() # process the timestamps file
    ff.process_folder() # process the folder containing the FreezeFrame data

# Run the main function
if __name__ == '__main__':
    main()