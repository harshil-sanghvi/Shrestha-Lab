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

A user can run the script by providing the path to the data folder and the path to the Excel file containing experiment details as arguments in the command line.

```python
python experiment_data_processing.py --path "C:\Users\user\Documents\Data" --exp_details_path "C:\Users\user\Documents\Experiment_Details.xlsx"
```

In this example, the script processes the data in the subfolders of the specified path and saves the processed data to Excel files. The experiment details are extracted from the Excel file provided and added to the processed data.

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
