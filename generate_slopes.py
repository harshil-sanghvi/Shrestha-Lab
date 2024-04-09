import os
import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm

class AvoidanceAnalyzer:
    def __init__(self, path):
        self.path = path
        self.avoidances = defaultdict(dict)
        self.slopes = defaultdict(dict)
        self.group_id = defaultdict(str)
        self.sex = defaultdict(str)
        self.percentage_saa_avoidance = defaultdict(str)
        self.average_latency = defaultdict(str)

    def extract_avoidances(self, file_path):
        """Extract avoidances from the given file."""
        df = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, sheet in df.items():
            if sheet_name.upper().startswith('SAA'):
                """Extract avoidances from the given sheet."""
                saa_df = sheet.iloc[2:, [1, 5]]
                av = saa_df['SAA#'].astype(int).values
                ids = saa_df['Animal ID'].values
                for i, j in zip(ids, av):
                    self.avoidances[i][sheet_name.upper()] = j

    def extract_group_id(self, file_path):
        """Extract group ID from the given file."""
        df = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, sheet in df.items():
            sheet = sheet.iloc[2:, [1, 4]]
            animal_ids = sheet['Animal ID'].values
            group = sheet['Group '].values
            for i, j in zip(animal_ids, group):
                self.group_id[i] = j

    def extract_sex(self, file_path):
        """Extract sex from the file."""
        df = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, sheet in df.items():
            sheet = sheet.iloc[2:, [1, 2]]
            animal_ids = sheet['Animal ID'].values
            sex = sheet['Sex'].values
            for i, j in zip(animal_ids, sex):
                self.sex[i] = j

    def extract_percentage_saa_avoidance(self, file_path):
        """Extract percentage SAA avoidance from the file."""
        df = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, sheet in df.items():
            if sheet_name.upper().startswith('LTM2'):
                sheet = sheet.iloc[2:, [1, 8]]
                animal_ids = sheet['Animal ID'].values
                percentage_saa_avoidance = sheet['SAA%'].values
                for i, j in zip(animal_ids, percentage_saa_avoidance):
                    self.percentage_saa_avoidance[i] = j
                break

    def extract_average_latency(self, file_path):
        """Extract average latency from the file."""
        df = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, sheet in df.items():
            if sheet_name.upper().startswith('LTM2'):
                sheet = sheet.iloc[2:, [1, -1]]
                animal_ids = sheet['Animal ID'].values
                average_latency = sheet['Average'].values
                for i, j in zip(animal_ids, average_latency):
                    self.average_latency[i] = j
                break

    def calculate_slopes(self):
        """Calculate slopes for each animal."""
        for animal_id, saa_slopes in tqdm(self.avoidances.items(), desc="Calculating slopes"):
            x = range(1, len(saa_slopes) + 1)
            y = list(saa_slopes.values())
            for i in range(1, len(x)):
                slope = (y[i] - y[i-1]) / (x[i] - x[i-1])
                self.slopes[animal_id][f"Slope_{i}"] = slope

    def process_files(self):
        """Process all files in the given path."""
        for root, dirs, files in os.walk(self.path):
            for file in tqdm(files, desc=f'Processing files in {root.split("\\")[-1]}'):
                if file.endswith(".xlsx"):
                    file_path = os.path.join(root, file)
                    self.extract_avoidances(file_path)
                    self.extract_group_id(file_path)
                    self.extract_sex(file_path)
                    self.extract_percentage_saa_avoidance(file_path)
                    self.extract_average_latency(file_path)

    def align_center(self, x):
        """Align text in cells to center."""
        return ['text-align: center' for _ in x]

    def save_data(self):
        """Save slopes and other data to Excel file."""
        output_file = "slopes_data.xlsx"
        writer = pd.ExcelWriter(output_file)

        df = pd.DataFrame(self.slopes).T
        df.columns = [f"SAA{i} vs SAA{i+1}" for i in range(1, 3)]

        df['Group ID'] = df.index.map(self.group_id)
        df['Sex'] = df.index.map(self.sex)
        df['% SAA Avoidance'] = df.index.map(self.percentage_saa_avoidance)
        df['Average Latency'] = df.index.map(self.average_latency)

        writer.sheets['Sheet1'] = writer.book.add_worksheet('Sheet1')
        
        for col in df.columns:
            max_len = max(df[col].astype(str).map(len).max(), len(col))
            col_idx = df.columns.get_loc(col)
            writer.sheets['Sheet1'].set_column(col_idx+1, col_idx+1, max_len + 1)

        df.index.name = 'Animal ID'
        df.style.apply(self.align_center, axis=0).to_excel(writer, index=True)
        writer.close()

if __name__ == "__main__":
    arg = ArgumentParser()
    arg.add_argument("--path", help="Path to the data folder", type=str, required=True)
    args = arg.parse_args()

    analyzer = AvoidanceAnalyzer(args.path)

    analyzer.process_files()
    analyzer.calculate_slopes()
    analyzer.save_data()