import ptc_ff
import pandas as pd

class StaggeredFF(ptc_ff.FreezeFrame):
    def __init__(self, staggered_timestamps_path, *args, **kwargs):
        super(StaggeredFF, self).__init__(*args, **kwargs)
        self.staggered_timestamps_path = staggered_timestamps_path
        self.staggered_timestamps = {}

    def process_staggered_timestamps(self):
        df = pd.read_excel(self.staggered_timestamps_path, usecols=[0, 1, 2], names=['Experiment', 'Animal ID', 'Start Delay'])

        # Filter rows without converting to string type, and drop NaN rows efficiently
        df = df[~df['Experiment'].str.contains('SN', na=False)].dropna(how='all')

        current_key = None
        for row in df.itertuples(index=False):
            experiment, animal_id, start_delay = row

            if isinstance(experiment, str):
                current_key = experiment.lower()
                if current_key not in self.staggered_timestamps:
                    self.staggered_timestamps[current_key] = {}
            elif current_key:
                self.staggered_timestamps[current_key][animal_id] = start_delay

    def get_ff_avg(self, animal_id, start, end, ff_df):
        start_delay = self.staggered_timestamps[self.experiment_name][animal_id]
        return super().get_ff_avg(animal_id, start_delay, end, ff_df)

ff = StaggeredFF(staggered_timestamps_path='/Users/harshil/Library/CloudStorage/GoogleDrive-Harshil.Sanghvi@stonybrook.edu/Shared drives/NBB_ShresthaLab_SharedDrive/LM - Harshil Sanghvi/New Data_ready for code/PTC_Staggered/PL_Rc_RmEngram.O4E PTC Freezeframe/20240829_PL_Rc_RmEngram.O4E CT5 PTC/20240829_PL_RC_RmEngram.O4E CT5 PTC Start_timestamps.xlsx', output_path='./', ct_path='/Users/harshil/Library/CloudStorage/GoogleDrive-Harshil.Sanghvi@stonybrook.edu/Shared drives/NBB_ShresthaLab_SharedDrive/LM - Harshil Sanghvi/New Data_ready for code/PTC_Staggered/PL_Rc_RmEngram.O4E PTC Freezeframe/PL_Rc_RmEngram.O4E Cohorts.xlsx', folder_path='/Users/harshil/Library/CloudStorage/GoogleDrive-Harshil.Sanghvi@stonybrook.edu/Shared drives/NBB_ShresthaLab_SharedDrive/LM - Harshil Sanghvi/New Data_ready for code/PTC_Staggered/PL_Rc_RmEngram.O4E PTC Freezeframe', timestamps_path='/Users/harshil/Library/CloudStorage/GoogleDrive-Harshil.Sanghvi@stonybrook.edu/Shared drives/NBB_ShresthaLab_SharedDrive/LM - Harshil Sanghvi/New Data_ready for code/PTC_Staggered/PTC Timestamps.xlsx')
ff.process_staggered_timestamps()
ff.process_timestamps()
ff.process_folder()
# print(ff.staggered_timestamps)