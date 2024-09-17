import pandas as pd
import re
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Precompile the regex for floating point numbers
FLOAT_PATTERN = re.compile(r'\d+\.\d+')

def determine_experiment_type(line: str) -> str:
    """
    Determine the type of experiment based on keywords in the line.
    """
    line_lower = line.lower()
    if 'ltm28d' in line_lower:
        return 'LTM28d'
    elif 'ltm14d' in line_lower:
        return 'LTM14d'
    elif 'ltm1' in line_lower:
        return 'LTM1'
    return 'Training'

def process_line(line: str) -> tuple:
    """
    Process a single line to extract the mouse name and CS values.
    """
    # Split the line and extract the mouse name    
    mouse_name = line.split('_')[0].split('-')[0]
    
    # Find all CS values and round them to 2 decimal places
    cs_values = [round(float(cs), 2) for cs in FLOAT_PATTERN.findall(line)]

    return mouse_name, cs_values

def read_data(file_path: str) -> dict:
    """
    Read the data from a file and categorize it into sheets.
    """
    data_sheets = {
        "LTM1": [],
        "LTM14d": [],
        "LTM28d": [],
        "Training": []
    }

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in tqdm(lines, desc="Processing lines"):
                experiment = determine_experiment_type(line)
                mouse_name, cs_values = process_line(line)
                row_data = [mouse_name] + cs_values
                data_sheets[experiment].append(row_data)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

    return data_sheets

def write_to_excel(data_sheets: dict, output_path: str):
    """
    Write the data to an Excel file with multiple sheets.
    """
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sheet_name, data in tqdm(data_sheets.items(), desc="Writing to Excel"):
            if data:
                num_cs = max(len(row) - 1 for row in data)
                column_names = ["Mouse Name"] + [f"CS{i+1}" for i in range(num_cs)]
                df = pd.DataFrame(data, columns=column_names)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    logging.info(f"Excel file created successfully at {output_path}")

def main():
    # Take input and output file paths from the user
    input_file = input("Enter the path to the input text file: ")
    output_file = input("Enter the path for the output Excel file: ")
    
    data_sheets = read_data(input_file)
    write_to_excel(data_sheets, output_file)

if __name__ == "__main__":
    main()