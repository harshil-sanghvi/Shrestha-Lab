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

    def align_center(self, x):
        """Align text in cells to center."""
        return ['text-align: center' for _ in x]

    def save_slopes(self):
        """Save slopes to Excel file."""
        output_file = "slopes.xlsx"
        writer = pd.ExcelWriter(output_file)

        df = pd.DataFrame(self.slopes).T
        df.columns = [f"SAA{i} vs SAA{i+1}" for i in range(1, 3)]

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
    analyzer.save_slopes()