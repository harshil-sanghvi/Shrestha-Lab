# Shrestha Lab

## Overview

This code is designed to process data files related to a specific experiment and generate an Excel file containing analyzed data. The script handles file parsing, data extraction, and Excel file creation. It also includes functions to add animal details to the generated Excel file based on a separate data source.

## Dependencies

The code relies on several Python libraries:

- `os`: For handling file paths and directories.
- `pandas` (`pd`): Used for data manipulation and Excel file creation.
- `numpy` (`np`): Provides support for numerical operations.
- `ExcelWriter` from `pandas`: Enables writing data to Excel files.
- `warnings`: Suppresses warning messages during data processing.
- `tqdm`: Used for progress tracking during file processing.

## Functions

1. **`check_saa_in_path(path)`**
   - Checks if the string 'SAA' is present in the given file path.

2. **`process_file(file_path, total_trials, col)`**
   - Processes an individual data file, extracts required information, and returns a DataFrame.

3. **`process_gs_data(GS_DIR_PATH)`**
   - Processes data from files within the specified directory (`GS_DIR_PATH`) and returns a consolidated DataFrame.

4. **`add_animal_details(data_df, exp_df, ct)`**
   - Adds animal details from a separate DataFrame (`exp_df`) to the main data DataFrame based on common identifiers.

5. **`align_center(x)`**
   - Helper function to align text in cells to the center for Excel styling.

6. **`process_and_save_data(PATH, exp_df, ct, add_animal_info=True)`**
   - Processes data in subfolders of a specified directory (`PATH`), adds animal details if required, and saves the analyzed data to Excel files.

## Usage

1. Define the paths to the main directory containing subfolders (`PATH`) and the file containing experiment details (`EXP_DETAILS_PATH`).

2. Read the experiment details into a DataFrame (`exp_df`) using `pd.read_excel`.

3. Loop through the subfolders in the main directory using `os.listdir(PATH)`.

4. Call `process_and_save_data()` for each subfolder to process the data and save the results to Excel files.

## Example

```python
PATH = r"MainDirectory"
EXP_DETAILS_PATH = r"ExperimentDetails.xlsx"

exp_df = pd.read_excel(EXP_DETAILS_PATH, usecols=[0, 1, 2, 3, 4])
exp_df.columns = ['SN', 'Animal', 'Sex', 'Subject ID', 'Group ']

for subfolder in tqdm(sorted(os.listdir(PATH)), desc="Processing subfolders", unit="folder"):
    ct = subfolder.split()[-1]
    GS_DIR_PATH = os.path.join(PATH, subfolder)
    try:
        if os.path.isdir(GS_DIR_PATH):
            process_and_save_data(GS_DIR_PATH, exp_df, ct)
    except Exception as e:
        print(f"Error processing {GS_DIR_PATH}: {e}")
```

In this example, replace "MainDirectory" with the actual path to the main directory containing subfolders, and "ExperimentDetails.xlsx" with the path to the file containing experiment details. The script processes each subfolder, adds animal details to the data, and saves the results as separate Excel files for each subfolder.

## Notes

- Ensure that the data files within subfolders follow the expected format for successful processing.
- Customize column names, file paths, and other parameters as per your specific data and requirements.

## Expected Folder Structure and Filename Format

### Folder Structure

The script expects a specific folder structure to process the data smoothly. The main directory should contain subfolders, each representing a different subset of data. Within each subfolder, the script expects to find a directory named "csv files" containing CSV data files.

- Main Directory
  - YYYYMMDD ExpName CohortID SAA 
    - SAA1
      - csv files
        - YYYY_MM_DD__HH_MM_SS_AnimalID.csv
        - YYYY_MM_DD__HH_MM_SS_AnimalID.csv
        - ...
  - YYYYMMDD ExpName CohortID SAA
    - LTM1
      - csv files
        - YYYY_MM_DD__HH_MM_SS_AnimalID.csv
        - YYYY_MM_DD__HH_MM_SS_AnimalID.csv
        - ...
  - ...

### Filename Format

For smooth processing, the script assumes that the CSV data files within each "csv files" directory follow a specific naming convention. The filename format should include relevant information such as animal ID, and experiment's date and time. Following is the format to be adhered:

- YYYY_MM_DD__HH_MM_SS_AnimalID.csv

Ensure that filenames are descriptive and include necessary identifiers to extract meaningful information during processing.
