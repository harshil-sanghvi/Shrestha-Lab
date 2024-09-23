import ptc_ff
import pandas as pd
import argparse

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
            experiment, animal_id, delay = row

            if isinstance(experiment, str):
                current_key = experiment.lower()
                if current_key not in self.staggered_timestamps:
                    self.staggered_timestamps[current_key] = {}
            elif current_key:
                self.staggered_timestamps[current_key][animal_id] = delay

    def get_ff_avg(self, animal_id, start, end, ff_df):
        delay = self.staggered_timestamps[self.experiment_name][animal_id]
        return super().get_ff_avg(animal_id, start + delay, end + delay, ff_df)

def main():
    '''Function to parse the command line arguments and process the FreezeFrame data.'''
    parser = argparse.ArgumentParser(description='Process FreezeFrame data')
    parser.add_argument('--timestamps', type=str, required=True, help='Path to the timestamps file')
    parser.add_argument('--ct', type=str, required=True, help='Path to the CT file')
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder containing the FreezeFrame data')
    parser.add_argument('--output', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--staggered_timestamps', type=str, required=True, help='Path to the staggered timestamps file')
    args = parser.parse_args()
    timestamps_path = args.timestamps
    ct_path = args.ct
    folder_path = args.folder
    output_path = args.output
    staggered_timestamps_path = args.staggered_timestamps

    ff = StaggeredFF(staggered_timestamps_path, timestamps_path, ct_path, folder_path, output_path) # create staggered FF object
    ff.process_staggered_timestamps() # process the staggered timestamps file
    ff.process_timestamps() # process the timestamps file
    ff.process_folder() # process the folder containing the FreezeFrame data

# Run the main function
if __name__ == '__main__':
    main()