import os
import pandas as pd
from tqdm import tqdm
import argparse

def gen_summary(path):
    """Generate summary Excel file from multiple Excel files."""
    files = os.listdir(path) # Get all files in the folder
    files = [f for f in files if f.endswith('.xlsx') and f != 'Summary.xlsx'] # Filter out non-Excel files
    data = {} # Dictionary to store data from each sheet in each Excel file
    for file in tqdm(files, desc='Processing files', unit='file'): # Loop through each Excel file
        file_path = os.path.join(path, file) # Get the full path of the file
        xls = pd.ExcelFile(file_path) # Read the Excel file
        sheet_names = xls.sheet_names # Get the names of all sheets in the Excel file
        for sheet_name in sheet_names: # Loop through each sheet in the Excel file
            df = pd.read_excel(xls, sheet_name, header=[0, 1]) # Read the sheet
            df = df.iloc[:, 1:] # Remove the first column
            sheet_name = sheet_name.upper() # Convert sheet name to uppercase
            data[sheet_name] = data.get(sheet_name, pd.DataFrame()) # Initialize an empty DataFrame if the sheet name is not in the dictionary
            data[sheet_name] = pd.concat([data[sheet_name], df], axis=0) # Concatenate the data from the sheet to the DataFrame in the dictionary

    with pd.ExcelWriter(os.path.join(path, 'Summary.xlsx')) as writer: # Write the data to a new Excel file
        for sheet_name, df in data.items(): # Loop through each sheet in the dictionary
            modified_columns = [((' ' if col[0].startswith('Unnamed') else col[0]),  
                                (' ' if col[1].startswith('Unnamed') else col[1])) for col in df.columns] # Modify column names
            df.columns = pd.MultiIndex.from_tuples(modified_columns) # Set the modified column names
            df = df.dropna(how='all') # Remove rows with all NaN values
            
            def align_center(x):
                """Align text in cells to center."""
                return ['text-align: center' for _ in x]
            
            df.reset_index(drop=True, inplace=True) # Reset the index
            df.index += 1 # Start the index from 1
            df.style.apply(align_center, axis=0).to_excel(writer, sheet_name=sheet_name, index=True) # Write the DataFrame to the Excel file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate summary Excel file from multiple Excel files.') # Create an argument parser
    parser.add_argument('path', type=str, help='Path to the folder containing Excel files.') # Add an argument for the path to the folder
    args = parser.parse_args() # Parse the arguments
    gen_summary(args.path) # Generate the summary Excel file