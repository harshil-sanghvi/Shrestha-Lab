import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings("ignore")

class FreezeFrame:
    def __init__(self, timestamps_path, ct_path, folder_path, output_path):
        self.timestamps_path = timestamps_path
        self.ct_path = ct_path
        self.folder_path = folder_path
        self.output_path = output_path
        self.training_timestamps, self.ltm_timestamps = None, None
        self.output = self.output_path
        self.cols = pd.MultiIndex.from_arrays([['Animal ID', '', ''] + ['CS+']*6 + ['CS-']*6 + [''] + ['ITI']*9 + ['']*2,
                                          ['', 'Threshold', 'Pre-CS'] + [str(i) for i in range(1, 6)] + ['Mean CS+'] + [str(i) for i in range(1, 6)] + ['Mean CS-'] + ['DI'] + [str(i) for i in range(1, 10)] + ['Mean ITI', 'Post-CS']])
        
    def process_timestamps(self):
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

        return new_df

    def process_folder(self):
        subfolders = [f for f in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, f))]
        for subfolder in subfolders:
            if 'CT1' in subfolder:
                continue
            self.process_subfolder(subfolder)

    def process_subfolder(self, subfolder):
        output_path = os.path.join(self.output, subfolder + '.xlsx')
        writer = pd.ExcelWriter(output_path)
        for file in os.listdir(os.path.join(self.folder_path, subfolder)):
            sheet_name = file.split('\\')[-1].split('.')[-2]
            if file.endswith('.csv'):
                file_path = os.path.join(self.folder_path, subfolder, file)
                data = self.process_file(file_path, sheet_name)
                data.to_excel(output_path, sheet_name=sheet_name, index=True)
        writer.close()        

    def clean_columns(self, columns):
        # Remove any leading or trailing whitespaces
        columns = [col.strip() for col in columns]
        # convert all columns to integers if possible
        columns = [int(float(column)) if column.replace('.', '').replace('-', '').replace('+', '').replace('e', '').isdigit() else column for column in columns]

        # print(columns)
        return columns

    def process_file(self, file_path, experiment_name):
        ff_df = pd.read_csv(file_path, header=1)
        # print(type(ff_df.columns))
        ff_df.columns = self.clean_columns(list(ff_df.columns))
        # print(type(ff_df.columns))
        # print(ff_df.columns)
        if 'LTM' in experiment_name:
            return self.process_ltm(ff_df)
        return self.process_training(ff_df)
    
    def process_training(self, ff_df):
        return self.process_data(ff_df, self.training_timestamps)
    
    def process_ltm(self, ff_df):
        return self.process_data(ff_df, self.ltm_timestamps)
    
    def process_data(self, ff_df, timestamps):
        df = pd.DataFrame(columns=self.cols)
        pre_cs_start, pre_cs_end = self.extract_timestamps(timestamps, 'Pre-CS')
        cs_plus_start, cs_plus_end = self.extract_timestamps(timestamps, r'CS\+')
        cs_minus_start, cs_minus_end = self.extract_timestamps(timestamps, r'CS\-')
        iti_start, iti_end = self.extract_timestamps(timestamps, 'ITI')
        post_cs_start, post_cs_end = self.extract_timestamps(timestamps, 'Post-CS')

        for animal_id in ff_df.iloc[1:, 0]:
            threshold = ff_df[ff_df.iloc[:, 0].astype(str).str.contains(str(animal_id))].loc[:, 'Threshold'].values[0]
            pre_cs = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(pre_cs_start, pre_cs_end)]
            cs_plus = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(cs_plus_start, cs_plus_end)]
            cs_minus = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(cs_minus_start, cs_minus_end)]
            iti = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(iti_start, iti_end)]
            post_cs = [self.get_ff_avg(animal_id, start, end, ff_df) for start, end in zip(post_cs_start, post_cs_end)]
            mean_cs_plus = np.mean(cs_plus)
            mean_cs_minus = np.mean(cs_minus)
            di = self.calculate_di(mean_cs_plus, mean_cs_minus)
            mean_iti = np.mean(iti)

            data = [animal_id, threshold, pre_cs[0], *cs_plus, mean_cs_plus, *cs_minus, mean_cs_minus, di, *iti, mean_iti, *post_cs]
            df = pd.concat([df, pd.DataFrame([data], columns=self.cols)], ignore_index=True)

        return df

    def calculate_di(self, mean_cs_plus, mean_cs_minus):
        try:
            return round((mean_cs_plus - mean_cs_minus) / (mean_cs_plus + mean_cs_minus), 1)
        except ZeroDivisionError:
            return 'N/A'
        
    def extract_timestamps(self, timestamps, label):
        start = timestamps[timestamps['Epoch'].str.contains(label)]['Onset'].values
        end = timestamps[timestamps['Epoch'].str.contains(label)]['Offset'].values
        return start, end

    def get_ff_avg(self, animal_id, start, end, ff_df):
        # return 0
        try:
            return float(ff_df[ff_df.iloc[:, 0].astype(str).str.contains(animal_id)].loc[:, [start]].mean(axis=1))
        except Exception as e:
            print(animal_id)
            # print(e)
            return 0

timestamps_path = r"G:\Shared drives\NBB_ShresthaLab_SharedDrive\LM - Harshil Sanghvi\PL_BLA.4EKD DTC Freezeframe\DTC Timestamps.xlsx"
ct_path = r"G:\Shared drives\NBB_ShresthaLab_SharedDrive\LM - Harshil Sanghvi\PL_BLA.4EKD DTC Freezeframe\PL_BLA.4EKD DTC Cohorts.xlsx"
folder_path = r"G:\Shared drives\NBB_ShresthaLab_SharedDrive\LM - Harshil Sanghvi\PL_BLA.4EKD DTC Freezeframe"
output_path = r"C:\Users\hsanghvi\OneDrive - Stony Brook University\Documents\Shrestha Lab\FF Results"

ff = FreezeFrame(timestamps_path, ct_path, folder_path, output_path)
ff.process_timestamps()
ff.process_folder()
