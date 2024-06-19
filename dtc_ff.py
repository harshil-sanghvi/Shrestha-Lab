import pandas as pd
import numpy as np
import os
from pandas import ExcelWriter
import warnings
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
        self.cols = None

    def getLTMcols(self):
        return pd.MultiIndex.from_arrays([['Animal ID', ' ', ' '] + ['CS+']*6 + ['CS-']*6 + [' '] + ['Post-CS+ ITI']*5  + ['Post-CS- ITI']*6 + [' ']*2,
                                          ['', 'Threshold', 'Pre-CS'] + [str(i) for i in range(1, 6)] + ['Mean CS+'] + [str(i) for i in range(1, 6)] + ['Mean CS-'] + ['D.I.'] + [str(i) for i in range(1, 5)] + ['Mean Post-CS+ ITI'] + [str(i) for i in range(1, 6)] + ['Mean Post-CS- ITI'] + ['Mean ITI', 'Post-CS']])
    
    def getTrainingCols(self):
        return pd.MultiIndex.from_arrays([['Animal ID', ' ', ' '] + ['CS+']*6 + ['CS-']*6 + [' '] + ['Post-CS+ ITI']*6  + ['Post-CS- ITI']*5 + [' ']*2,
                                          ['', 'Threshold', 'Pre-CS'] + [str(i) for i in range(1, 6)] + ['Mean CS+'] + [str(i) for i in range(1, 6)] + ['Mean CS-'] + ['D.I.'] + [str(i) for i in range(1, 6)] + ['Mean Post-CS+ ITI'] + [str(i) for i in range(1, 5)] + ['Mean Post-CS- ITI'] + ['Mean ITI', 'Post-CS']])
        
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
            sheet_name = file.split('\\')[-1].split('.')[-2] # extract the sheet name
            if file.endswith('.csv'): # if the file is a CSV file
                file_path = os.path.join(self.folder_path, subfolder, file) # set the file path
                data = self.process_file(file_path, sheet_name) # process the FreezeFrame data
                final = pd.merge(self.ct_df, data, on='Animal ID', how='inner') # merge the cohort data with the FreezeFrame data
                final.style.apply(self.align_center, axis=0).to_excel(writer, sheet_name=sheet_name, index=True) # write the data to the Excel file
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
        if 'LTM' in experiment_name: # if the experiment is LTM
            return self.process_ltm(ff_df) # process the FreezeFrame data for LTM
        return self.process_training(ff_df) # process the FreezeFrame data for training
    
    def process_training(self, ff_df):
        '''Function to process the FreezeFrame data for the training experiment.'''
        self.exp = 'Training'
        return self.process_data(ff_df, self.training_timestamps, 'Training')
      
    def process_ltm(self, ff_df):
        '''Function to process the FreezeFrame data for the LTM experiment.'''
        self.exp = 'LTM'
        return self.process_data(ff_df, self.ltm_timestamps, 'LTM')
    
    def process_data(self, ff_df, timestamps, exp):
        '''Function to process the FreezeFrame data for the given timestamps.'''
        if exp == 'Training':
            self.cols = self.getTrainingCols()
        else:
            self.cols = self.getLTMcols()

        df = pd.DataFrame(columns=self.cols)

        pre_cs_start, pre_cs_end = self.extract_timestamps(timestamps, 'Pre-CS') # extract the start and end timestamps for Pre-CS
        cs_plus_start, cs_plus_end = self.extract_timestamps(timestamps, r'CS+') # extract the start and end timestamps for CS+
        cs_minus_start, cs_minus_end = self.extract_timestamps(timestamps, r'CS-') # extract the start and end timestamps for CS-
        post_cs_plus_iti_start, post_cs_plus_iti_end = self.extract_timestamps(timestamps, 'Post-CS+ ITI') # extract the start and end timestamps for Post-CS+ ITI
        post_cs_minus_iti_start, post_cs_minus_iti_end = self.extract_timestamps(timestamps, 'Post-CS- ITI') # extract the start and end timestamps for Post-CS- ITI
        post_cs_start, post_cs_end = self.extract_timestamps(timestamps, 'Post-CS') # extract the start and end timestamps for Post-CS
        
        for animal_id in ff_df.iloc[1:, 0]: # for each animal ID
            if str(animal_id) == 'nan': # if the animal ID is 'nan', skip it
                continue
            threshold = ff_df[ff_df.iloc[:, 0].astype(str).str.contains(str(animal_id))].loc[:, 'Threshold'].values[0] # extract the threshold
            pre_cs = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(pre_cs_start, pre_cs_end)] # extract the average of the FreezeFrame data for Pre-CS
            cs_plus = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(cs_plus_start, cs_plus_end)] # extract the average of the FreezeFrame data for CS+
            cs_minus = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(cs_minus_start, cs_minus_end)] # extract the average of the FreezeFrame data for CS-
            post_cs_plus_iti = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(post_cs_plus_iti_start, post_cs_plus_iti_end)] # extract the average of the FreezeFrame data for Post-CS+ ITI
            post_cs_minus_iti = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(post_cs_minus_iti_start, post_cs_minus_iti_end)] # extract the average of the FreezeFrame data for Post-CS- ITI
            post_cs = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(post_cs_start, post_cs_end)] # extract the average of the FreezeFrame data for Post-CS
            mean_cs_plus = round(np.mean(cs_plus), 2) # calculate the mean of the CS+ data
            mean_cs_minus = round(np.mean(cs_minus), 2) # calculate the mean of the CS- data
            di = self.calculate_di(mean_cs_plus, mean_cs_minus) # calculate the D.I.
            mean_post_cs_plus_iti = round(np.mean(post_cs_plus_iti), 2) # calculate the mean of the Post-CS+ ITI data
            mean_post_cs_minus_iti = round(np.mean(post_cs_minus_iti), 2) # calculate the mean of the Post-CS- ITI data
            mean_iti = round(np.sum([*post_cs_plus_iti, *post_cs_minus_iti]) / len([*post_cs_plus_iti, *post_cs_minus_iti]), 2) # calculate the mean of the ITI data

            data = [animal_id.split()[-1], threshold, pre_cs[0], *cs_plus, mean_cs_plus, *cs_minus, mean_cs_minus, di, *post_cs_plus_iti, mean_post_cs_plus_iti, *post_cs_minus_iti, mean_post_cs_minus_iti, mean_iti, post_cs[-1]]
            
            df = pd.concat([df, pd.DataFrame([data], columns=self.cols)], ignore_index=True) # concatenate the data to the DataFrame
        return df # return the DataFrame

    def calculate_di(self, mean_cs_plus, mean_cs_minus):
        '''Function to calculate the D.I.'''
        try: # try to calculate the D.I.
            return round((mean_cs_plus - mean_cs_minus) / (mean_cs_plus + mean_cs_minus), 2) # return the D.I.
        except ZeroDivisionError: # if there is a ZeroDivisionError, return 'N/A'
            return 'N/A'
        
    def extract_timestamps(self, timestamps, label):
        '''Function to extract the start and end timestamps for the given label.'''
        start = timestamps[timestamps['Epoch'].str.startswith(label)]['Onset'].values # extract the start timestamps 
        end = timestamps[timestamps['Epoch'].str.startswith(label)]['Offset'].values # extract the end timestamps
        return start, end # return the start and end timestamps

    def get_ff_avg(self, animal_id, start, end, ff_df):
        '''Function to calculate the average of the FreezeFrame data for the given animal ID for the given start and end timestamps.'''
        try:
            sub_df = ff_df.loc[ff_df.iloc[:, 0].astype(str).str.contains(animal_id), start:end] # extract the FreezeFrame data for the given animal ID and timestamps
            sub_df = sub_df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x).replace("NaN", 0).apply(pd.to_numeric, errors='coerce') # clean the data by first stripping the strings, replacing "NaN" with 0, and converting the data to numeric
            return float(sub_df.mean(axis=1).round(2)) # return the average of the data
        except Exception as e: # if there is an exception, return 0
            if e.args[0].startswith("cannot convert the series to "):
                print(f'Multiple values for {animal_id}')
            elif e.args[0].startswith("cannot do slice indexing"):
                print(f'No values for {animal_id} for timestamp value {e.args[0].split()[-4]}')
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

    ff = FreezeFrame(args.timestamps, args.ct, args.folder, args.output)
    ff.process_timestamps()
    ff.process_folder()

# Run the main function
if __name__ == '__main__':
    main()