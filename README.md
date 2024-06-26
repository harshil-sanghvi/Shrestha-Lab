**Manual: Shrestha Lab Python Scripts**

**Harshil Sanghvi & Prerana Shrestha**

_Last Updated 06/21/2024_

**A. Python Installation and GitHub Repository**

To facilitate data processing from Graphic State and FreezeFrame experiments, Python and essential libraries are required. Below are the detailed steps for Python installation and instructions for accessing the GitHub repository.

1. **Python Installation:**
   - **Mac:** Follow [this YouTube tutorial](https://youtu.be/nhv82tvFfkM) until 3:44.
   - **Windows:** Follow [this YouTube tutorial](https://youtu.be/ERcsRnUQ64s) until 2:31.
   - Open Terminal (Mac) or Command Prompt (Windows).
   - Install required packages using pip:
     ```
     pip install pandas numpy argparse Jinja2
     ```

2. **GitHub Repository:**
   - Access the Shrestha Lab Python scripts repository [here](https://github.com/harshil-sanghvi/Shrestha-Lab).

**B. Pavlovian Threat Conditioning - FreezeFrame (FF) Data**

For processing data from FreezeFrame experiments with discrete CS epochs:

- **Data Sources:**
  - Cohort details are sourced from `cohorts.xlsx`.
  - CS epoch timestamps are sourced from `PTC Timestamps.xlsx` or `DTC Timestamps.xlsx`.

- **Protocol and Folder Structure:**

	-   **Protocol:**  Refer to the detailed protocol for exporting and preprocessing raw data from FreezeFrame available  [here](https://docs.google.com/document/d/1-GL7XAA1Yo-S_kxhXourc54XA5-0dqa0jA3sxfy8ZHM/edit#heading=h.r7fsm9fq7bbj).
	-   **Cohort Details and Timestamps:**  Ensure  `cohorts.xls`  and   `PTC Timestamps.xlsx`  or  `DTC Timestamps.xlsx` follow the structure outlined in the protocol to facilitate smooth data processing.

	- **Folder Structure:**
  ```
  - Parent Experiment Folder (e.g., PL_CamK2a.4EKD PTC Freezeframe)
    - Child Experiment CT1 Subfolder (e.g., 20220322 PL_CamK2a.4EKD CT1 PTC)
      - freeze_SAA1.csv
      - freeze_SAA2.csv
      - ...
    - Child Experiment CT2 Subfolder
    - ...
  ```

- **Scripts and Usage:**
  - `ptc_ff.py` for PTC data (CS+ only).
  - `dtc_ff.py` for DTC data (CS+ and CS-).
  - `dtc_unp_ff.py` for modified DTC protocols.

  **Usage:**
  ```
  python filename.py --timestamps "/path/to/timestamps.xlsx" --ct "/path/to/cohorts.xlsx" --folder "/path/to/freezeframe_data" --output "/path/to/output_folder"
  ```

**C. Signaled Active Avoidance - Graphic State (GS) Data**

For processing data from Graphic State experiments:

- **Data Sources:**
  - Cohort details are sourced from `cohorts.xlsx`.

- **Protocol and Folder Structure:**
	-   **Protocol:**  Refer to the detailed protocol for exporting and preprocessing raw data from Graphic State available  [here](https://docs.google.com/document/d/17RiWy8IkbLBCMEBfHGFb2WbEahipzEGvwAm_slDDRR0/edit#heading=h.ra1nhlil3bl7).
	-   **Cohort Details:**  Ensure  `cohorts.xls`  follow the structure outlined in the protocol to facilitate smooth data processing.

	- **Folder Structure:**
  ```
  - Parent Experiment Folder (e.g., PL_SIStag.TSC SAA)
    - Date ExperimentName CT1 SAA
      - SAA1
        - csv files
          - 2023_02_28__16_12_00_A327.csv
          - ...
      - SAA2
        - csv files
          - ...
    - Date ExperimentName CT1 SAA
    - ...
  ```

- **Scripts and Usage:**
  - `saa_gs.py` for GS data (CS+ only).
  - `dsaa_gs.py` for GS data (CS+ and CS-).

  **Usage:**
  ```
  python filename.py --path "/path/to/main_folder" --exp_details_path "/path/to/cohorts.xlsx"
  ```

**D. Signaled Active Avoidance - FreezeFrame (FF) Data**

For processing data combining GS and FF experiments:

- **Data Sources:**
  - **Cohort Information:** Utilize `cohorts.xlsx` containing experiment details per cohort and animal, adhering to protocol specifications.
  - **CS Timestamps:** Access the folder containing CSV files of CS timestamps generated from `saa_gs.py` or `dsaa_gs.py`.
	  - Ensure the folder structure for GS processed data follows:

```
- Parent experiment folder (e.g., PL_SIStag.TSC SAA)
    - ExperimentName CT1 Date.xlsx (e.g., PL SISTAG.TSC CT2 20230326.xlsx)
    - ...
 ```

- **Folder Structure:**
  - Maintain consistency with the structure outlined in Section B for seamless integration and data retrieval.

- **Scripts and Usage:**
  - `saa_ff.py` for FF data (CS+ only).
  - `dsaa_ff.py` for FF data (CS+ and CS-).

  **Usage:**
   - Execute the scripts using the same procedure as described in Section B for FreezeFrame experiments.


**Conclusion**

These scripts are designed to streamline data processing from both Graphic State and FreezeFrame experiments conducted by the Shrestha Lab. By following the specified protocols, folder structures, and script usage guidelines, lab members can efficiently analyze and interpret experimental results. For further details, refer to the provided protocols and documentation links.
