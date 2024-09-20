import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
import tdt
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from scipy.stats import sem
import seaborn as sns
import os
import warnings
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Adjust the logging level as needed
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fib_photo.log'),  # Log to a file
        logging.StreamHandler()  # Also output to the console
    ]
)

# Ignore warnings
warnings.filterwarnings("ignore")

# Precompile the regex for floating point numbers
FLOAT_PATTERN = re.compile(r'\d+\.\d+')

class Mouse:
    def __init__(self, 
                 file_path: str, 
                 isTrain: bool, 
                 PRE_TIME: int, 
                 POST_TIME: int, 
                 signal: str = "_465A", 
                 control: str = "_405A", 
                 isReins: bool = False,
                 aucLogPath: str = 'auc.txt'):
        """
        A class for representing mouse data and performing analysis.

        Parameters:
        - file_path (str): The path to the data file.
        - isTrain (bool): Whether the data is for training or not.
        - PRE_TIME (int): The duration of the pre-stimulus period in seconds.
        - POST_TIME (int): The duration of the post-stimulus period in seconds.

        Attributes:
        - BLOCK_PATH (str): The path to the data file.

        The following attributes will be calculated from the BLOCK _PATH using calculate_properties()
        - data: Data loaded from the specified file.
        - time: Time vector for the data.
        - fs: Sampling frequency of the data.
        - isTrain (bool): Whether the data is for training or not.
        - CSon: Onset times of the conditioned stimulus (CS).
        - CSoff: Offset times of the conditioned stimulus (CS).
        - USon: Onset times of the unconditioned stimulus (US).
        - USoff: Offset times of the unconditioned stimulus (US).
        - samplingrate: Sampling rate of the data.
        - signal_lowpass: Low-pass filtered signal data.
        - control_lowpass: Low-pass filtered control data.
        - signal_doubleExpFit: Double exponential fitted signal data.
        - control_doubleExpFit: Double exponential fitted control data.
        - signal_doubleFittedCurve: Fitted curve for the signal data.
        - control_doubleFittedCurve: Fitted curve for the control data.
        - dFF: Delta F/F calculated from the signal and control data.
        - Y_fit_all: Fitted values for the linear fit subtraction.
        - Y_dF_all: Delta F values for the linear fit subtraction.
        - zScore: Z-score calculated from the delta F values.
        - dFF_snips: Delta F/F snippets.
        - PRE_TIME (int): The duration of the pre-stimulus period in seconds.
        - POST_TIME (int): The duration of the post-stimulus period in seconds.
        - peri_time: Time vector for the peri-stimulus period.

        Methods:
        - load_data(): Load data from the specified file.
        - calculate_properties(): Calculate various properties of the data.
        - lowpass_filter(): Apply low-pass filter to the signal and control data.
        - doubleFitSubtraction(): Perform double exponential fit subtraction.
        - linearFitSubtraction(): Perform linear fit subtraction.
        - calculate_dFF_snips(): Calculate delta F/F snippets.
        - plot_heat_map(): Plot a heatmap of the delta F/F snippets.
        - get_and_plot_AUC(): Calculate and plot area under the curve (AUC) values.
        """

        # File path and stimulus times
        self.BLOCK_PATH = file_path
        self.PRE_TIME = PRE_TIME
        self.POST_TIME = POST_TIME
        self.isTrain = isTrain
        self.isReins = isReins
        self.aucLogPath = aucLogPath
        
        # Initialize data-related attributes to None
        self.t1 = None
        self.data = self.time = self.fs = None
        self.CSon = self.CSoff = self.USon = self.USoff = None
        self.samplingrate = None
        self.signal_lowpass = self.control_lowpass = None
        self.signal_doubleExpFit = self.control_doubleExpFit = None
        self.signal_doubleFittedCurve = self.control_doubleFittedCurve = None
        self.dFF = self.Y_fit_all = self.Y_dF_all = self.zScore = None
        self.dFF_snips = self.peri_time = self.savgol_zscore = None

        # Load data and calculate properties
        print('-------- Loading Data --------')
        self.__load_data()

        print('-------- Calculating Properties --------')
        self.__calculate_properties(signal, control)

    #Private Methods
    def __load_data(self):
        """
        Load data from the specified file.

        Reads the data from the file specified by BLOCK_PATH and extracts relevant
        information such as signal, control, and onset/offset times of CS and US stimuli.
        This method sets the attribute self.data.

        Returns:
        None
        """
        # Load the full block of data initially
        data = tdt.read_block(self.BLOCK_PATH)

        # Determine the first time point
        self.t1 = data.epocs.PrtB.onset[0]

        # Determine time range based on training mode
        if self.isTrain:
            t2 = data.epocs.CS__.offset[1] + 1800
        else:
            t2 = 0  # No specific t2, load from t1 onwards

        # Load the relevant data based on the time range
        self.data = tdt.read_block(self.BLOCK_PATH, t1=self.t1, t2=t2)

    def __calculate_properties(self, signal, control):
        """
        Calculate various properties of the loaded data.

        Calculates properties such as sampling rate, time vector,
        onset and offset times of CS and US stimuli, and performs
        low-pass filtering, photobleaching correction (double-fit subtraction), 
        and motion correction (linear fit subtraction).

        Returns:
        None
        """
        # Fetching the signal and control data, ensuring same length
        self.signal = self.data.streams[signal].data
        self.control = self.data.streams[control].data
        max_len = min(len(self.signal), len(self.control))
        self.signal, self.control = self.signal[:max_len], self.control[:max_len]
        
        # Calculate the sampling rate and time vector
        self.fs = self.data.streams[signal].fs
        self.time = np.linspace(1, len(self.signal), len(self.signal)) / self.fs

        # Calculate CS and US onsets/offsets for training data
        if self.isTrain:
            self.USon = self.data['epocs']['Shck']['onset'] - self.t1
            self.USoff = self.data['epocs']['Shck']['offset'] - self.t1

        self.CSon = self.data['epocs']['CS__']['onset'] - self.t1
        self.CSoff = self.data['epocs']['CS__']['offset'] - self.t1

        # Perform low-pass filtering on the signal and control
        self.signal_lowpass, self.control_lowpass = self.__lowpass_filter(self.signal, self.control, self.fs)

        # Photobleaching correction (double exponential fit subtraction)
        self.signal_doubleExpFit, self.control_doubleExpFit, self.signal_doubleFittedCurve, self.control_doubleFittedCurve = (
            self.__doubleFitSubtraction(self.signal_lowpass, self.control_lowpass, self.time)
        )

        # Motion correction (linear fit subtraction)
        self.dFF, self.Y_dF_all, self.Y_fit_all = self.__linearFitSubtraction(
            self.signal_doubleExpFit, self.control_doubleExpFit, self.signal_doubleFittedCurve
        )

        # Calculate z-scores
        self.zScore = (self.Y_dF_all - np.mean(self.Y_dF_all)) / np.std(self.Y_dF_all)
        self.savgol_zscore = self.__get_savgol_zscore()

        # Calculate delta F/F snippets
        self.dFF_snips, self.peri_time = self.__calculate_dFF_snips(
            self.signal, self.savgol_zscore, self.fs, self.CSon, self.time, self.PRE_TIME, self.POST_TIME
        )

    def __lowpass_filter(self, signal, control, samplingrate):
        """
        Apply low-pass Butterworth filter to the signal and control data.

        Parameters:
        - signal (array): Signal data to be filtered.
        - control (array): Control data to be filtered.
        - samplingrate (int): Sampling rate of the data.

        Returns:
        Tuple containing the filtered signal and control data.
        """
        order = 3  # Using a lower filter order (6 // 2 = 3) for computational efficiency
        cutoff = 6
        nyquist_freq = samplingrate / 2  # Nyquist frequency
        normalized_cutoff = cutoff / nyquist_freq

        b, a = butter(order, normalized_cutoff, btype='low')
        filtered_signal = filtfilt(b, a, signal)
        filtered_control = filtfilt(b, a, control)

        return filtered_signal, filtered_control

    def __doubleFitSubtraction(self, signal, control, time):
        """
        Performs PHOTO BLEACHING CORRECTION or double exponential fit subtraction.

        Fits double exponential curves to both the signal and control data, and
        subtracts the fitted curves from the original data.

        Parameters:
        - signal (array): Signal data.
        - control (array): Control data.
        - time (array): Time vector.

        Returns:
        Tuple containing the signal and control data after subtraction,
        as well as the fitted curves for both signal and control data.
        """
        def double_exponential(x, a, b, c, d):
            return a * np.exp(b * x) + c * np.exp(d * x)

        # Ensure the inputs are numpy arrays of float32 type
        signal, control, time = map(lambda arr: np.array(arr, dtype=np.float32), [signal, control, time])

        # Perform curve fitting for both signal and control
        popt_signal, _ = curve_fit(double_exponential, time, signal, p0=[0.05, -0.05, 0.05, -0.05])
        popt_control, _ = curve_fit(double_exponential, time, control, p0=[0.05, -0.05, 0.05, -0.05])

        # Compute the fitted curves
        fitted_curve_signal = double_exponential(time, *popt_signal)
        fitted_curve_control = double_exponential(time, *popt_control)

        # Subtract fitted curves from the original data
        signal_corrected = signal - fitted_curve_signal
        control_corrected = control - fitted_curve_control

        return signal_corrected, control_corrected, fitted_curve_signal, fitted_curve_control

    def __linearFitSubtraction(self, signal, control, fitted_curve_signal):
        """
        Perform Motion Correction or linear fit subtraction.

        Performs linear fit subtraction between the signal and control data,
        then calculates delta F/F values using the fitted curve for the signal.

        Parameters:
        - signal (array): Signal data.
        - control (array): Control data.
        - fitted_curve_signal (array): Fitted curve for the signal data.

        Returns:
        Tuple containing the delta F/F values, fitted values, and delta F
        values for all time points.
        """
        # Perform linear polynomial fitting (1st degree)
        bls = np.polyfit(control, signal, 1)
        
        # Calculate the fitted and delta F values
        fitted_values = bls[0] * control + bls[1]
        delta_F_values = signal - fitted_values

        # Calculate delta F/F values
        epsilon = 1e-9  # Avoid division by zero
        dFF = 100 * delta_F_values / (fitted_curve_signal + epsilon)

        return dFF, delta_F_values, fitted_values


    def __calculate_dFF_snips(self, signal, zscore, fs, CSon, time, PRE_TIME, POST_TIME):
        """
        Calculate delta F/F snippets.

        Parameters:
        - signal (array): Signal data.
        - zscore (array): Z-score calculated from the signal data.
        - fs (int): Sampling frequency of the data.
        - CSon (array): Onset times of CS stimuli.
        - time (array): Time vector.
        - PRE_TIME (int): Pre-stimulus duration.
        - POST_TIME (int): Post-stimulus duration.

        Returns:
        Tuple containing the delta F/F snippets and corresponding peri-event time vector.
        """
        # Calculate sample range for pre and post stimulus durations
        trange = np.array([-PRE_TIME, POST_TIME]) * fs
        trange = trange.astype(int)
        
        dFF_snips = []
        
        # Precompute length of the time snippets to avoid recalculating in the loop
        snippet_length = trange[1] - trange[0]

        for on in CSon:
            if on >= PRE_TIME:  # Skip events occurring before PRE_TIME
                # Get the index of the CS onset in the time array
                onset_idx = np.searchsorted(time, on)

                # Extract the pre-stimulus and post-stimulus window
                pre_idx = onset_idx + trange[0]
                post_idx = onset_idx + trange[1]

                if pre_idx >= 0 and post_idx < len(zscore):  # Ensure indices are within bounds
                    dFF_snips.append(zscore[pre_idx:post_idx])
                else:
                    dFF_snips.append(np.zeros(snippet_length))  # Fill with zeros if out of bounds

        # Calculate the peri-event time vector
        peri_time = np.linspace(-PRE_TIME, POST_TIME, snippet_length)

        return np.array(dFF_snips), peri_time

    def __get_savgol_zscore(self):
        window_length = 6501
        polyorder = 3
        zscore_smoothed = savgol_filter(self.zScore, window_length, polyorder)
        return zscore_smoothed

    def plot_mean_response(self, ax, peri_time, mean_dFF_snips, sem_dFF_snips, mean_band_color):
        # Plot the mean response with SEM
        ax.plot(peri_time, mean_dFF_snips, linewidth=1, color=mean_band_color, label='Mean Response')
        ax.fill_between(peri_time, mean_dFF_snips + sem_dFF_snips, mean_dFF_snips - sem_dFF_snips,
                        facecolor=mean_band_color, alpha=0.3, label='SEM')

    def plot_stimulus_box(self, ax, xmin, xmax, ymin, ymax, alpha, color, label, offset_bar_color):
        ax.fill_between([xmin, xmax], ymin, ymax, color=color, alpha=alpha)
        ax.plot([xmin, xmax], [ymax, ymax], color=offset_bar_color, linewidth=2, label=label)

    def get_and_plot_PETH(self, filename, size_x=8, size_y=6):
        # Colors
        offset_cs_color = '#BB1F1F'
        offset_us_color = '#040404'
        cs_highlight_color = '#FBF887'
        mean_band_color = 'black'

        # Data variables
        dff_snips_temp = self.dFF_snips
        peri_time_temp = self.peri_time

        # Statistics for the dFF snips
        mean_dFF_snips = np.mean(dff_snips_temp, axis=0)
        sem_dFF_snips = sem(dff_snips_temp, axis=0)
        max_dFF_snips = np.max(dff_snips_temp)
        min_dFF_snips = np.min(dff_snips_temp)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(size_x, size_y))

        # Plot mean response and SEM
        self.plot_mean_response(ax, peri_time_temp, mean_dFF_snips, sem_dFF_snips, mean_band_color)

        # Calculate y-axis limits
        ymin = min_dFF_snips - 0.5
        ymax = max_dFF_snips + 0.5
        diff = ymax - ymin

        # Plot CS and US onset/offset
        self.plot_stimulus_box(ax, xmin=0, xmax=30, ymin=ymin, ymax=ymax, alpha=0.3, color=cs_highlight_color, 
                        label='CS', offset_bar_color=offset_cs_color)

        if self.isTrain:
            self.plot_stimulus_box(ax, xmin=28, xmax=30, ymin=ymin, ymax=ymax + (0.02 * diff), alpha=0, color=cs_highlight_color, 
                        label='US', offset_bar_color=offset_us_color)

        # Axis labels and title
        ax.set_xlabel('Seconds')
        ax.set_ylabel(r'$\Delta$F/F')
        ax.set_title('Peri-Event Trial Responses')
        ax.legend()

        # Save and show the plot
        self.save_plot(fig, 'PETH Histograms', filename)
        print('########## Saved PETH ##########')

    def log_auc(self, mousename, auc_values):
        with open(self.aucLogPath, 'a') as f:
            f.write(f'{mousename}: {auc_values}\n')

    def get_and_plot_AUC(self, filename):
        """
        Calculate and plot area under the curve (AUC) values for each CS event.

        Returns:
        AUC values and their corresponding standard errors for each event and time interval.
        """
        time_intervals = [(0, 30)]  # Pre-defined time intervals for AUC calculation
        num_events = len(self.dFF_snips)

        auc_values_cs = np.zeros(num_events)
        std_cs = np.zeros(num_events)

        # Loop through each event and calculate AUC for the specified time interval
        for i, dFF in enumerate(self.dFF_snips):
            for start, end in time_intervals:
                start_index = np.searchsorted(self.peri_time, start)
                end_index = np.searchsorted(self.peri_time, end)
                time_subset = self.peri_time[start_index:end_index + 1]

                # Calculate AUC using the trapezoid rule and standard deviation
                auc_values_cs[i] = trapezoid(dFF[start_index:end_index + 1], time_subset)
                std_cs[i] = np.std(dFF[start_index:end_index + 1])

        self.log_auc(filename, auc_values_cs)
        
        # Plotting AUC values
        bar_width = 0.8

        if self.isTrain:
            plt.figure(figsize=(8, 6))
            index = np.arange(1, num_events + 1)
            plt.bar(index, auc_values_cs, bar_width, capsize=5, label='CS', color='#BB1F1F')
            plt.xticks(index)
        else:
            plt.figure(figsize=(3, 5))
            # Calculate mean and SEM for CS events
            avg_auc_values_cs = np.mean(auc_values_cs)
            avg_std_cs = sem(auc_values_cs)
            plt.bar([0.5], avg_auc_values_cs, width=bar_width, yerr=avg_std_cs, capsize=5, label='CS', color='#BB1F1F')
            plt.xticks([0.5], ['1'])

        # Common formatting
        plt.xlabel("CS Event")
        plt.ylabel("AUC Values")
        plt.title("AUC for CS Events" if self.isTrain else "AUC with SEM - LTM")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        # Save the plot
        self.save_plot(plt, 'AUC Plots', filename)
        print('########## Saved AUC ##########')

        return auc_values_cs, std_cs

    def plot_heat_map(self, size_x, size_y, filename, title, ylabel=""):
        """
        Plot a heatmap of the delta F/F snippets using Seaborn.

        Plots a heatmap of the delta F/F snippets calculated for each CS event,
        with time on the x-axis and CS event on the y-axis.

        Parameters:
        - size_x (int): Width of the heatmap plot.
        - size_y (int): Height of the heatmap plot.
        - title (str): Title of the plot.
        - ylabel (str): Label for the y-axis (default is an empty string).

        Returns:
        None
        """

        # Calculate actual time values for x-axis
        time_range = np.linspace(-self.PRE_TIME, self.POST_TIME, len(self.dFF_snips[0]))

        # Define x-tick labels at 10-second intervals
        xtick_labels = np.arange(0, self.POST_TIME + 1, 10)

        # Create the figure and axis for the heatmap
        fig, ax = plt.subplots(figsize=(size_x, size_y))

        # Reverse the order of dFF_snips for display purposes
        dFF_snips_reversed = np.flip(self.dFF_snips, axis=0)

        # Generate the heatmap with Seaborn
        sns.heatmap(dFF_snips_reversed, cmap='jet', ax=ax, cbar=True)

        # Set labels and title
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Map the x-ticks to the appropriate time points
        xtick_positions = np.interp(xtick_labels, time_range, np.arange(len(self.dFF_snips[0])))
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)

        # Set the y-ticks and reverse the labels for correct CS event display
        ax.set_yticks(np.arange(len(self.dFF_snips)) + 0.5)
        ax.set_yticklabels(np.arange(1, len(self.dFF_snips) + 1)[::-1])

        # Set the axis limits
        ax.set_xlim(0, len(self.dFF_snips[0]) - 1)
        ax.set_ylim(0, len(self.dFF_snips))

        self.save_plot(fig, 'Heatmaps', filename)
        print('########## Saved Heatmap ##########')

    def setup_plot_style(self):
        plt.style.use('seaborn-v0_8-white')
        prop = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family="Arial")))
        return prop

    def determine_experiment_name(self, filename):
        if 'ltm28d' in filename.lower():
            return 'LTM28d'
        elif 'ltm14d' in filename.lower():
            return 'LTM14d'
        elif 'ltm1' in filename.lower():
            return 'LTM1'
        return 'Training'
    
    def save_plot(self, fig, output_dir, filename):
        animal_id = filename.split('_')[0].split('-')[0]
        experiment_name = self.determine_experiment_name(filename)
        
        output_dir_lower = output_dir.lower()
        file_suffix = ''

        if 'heatmap' in output_dir_lower:
            file_suffix = 'heatmap'
        elif 'auc' in output_dir_lower:
            file_suffix = 'AUC'
        elif 'peth' in output_dir_lower:
            file_suffix = 'PETH'
        elif 'dff' in output_dir_lower:
            file_suffix = 'dFF'
        elif 'zscores' in output_dir_lower:
            file_suffix = 'Zscores'

        if self.isTrain:
            filename = f'{animal_id}_train_{file_suffix}.png'
        else:
            filename = f'{animal_id}_{experiment_name}_{file_suffix}.png'

        output_dir = os.path.join(output_dir, animal_id)
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

    def plot_dFF_trace(self, last_index):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.time[:last_index], self.dFF[:last_index], linewidth=1, color='black', label='Signal', alpha=0.7)
        ax.set_ylabel(r"$\Delta$ F/F")
        ax.set_xlabel('Time(s)')
        return fig

    def plot_zscore_trace(self, prop):
        fig, ax = plt.subplots(figsize=(12, 4))
        time = np.linspace(1, len(self.signal), len(self.signal)) / self.data.streams['_465A'].fs
        ax.plot(time, self.zScore, linewidth=1, color='black', alpha=1)
        ax.set_ylabel(r"Z-score", fontproperties=prop)
        ax.set_xlabel('Time(s)', fontproperties=prop)
        ax.set_title('Z-score', fontproperties=prop)
        return fig, ax

    def plot_pair(self, xmin, xmax, ax, alpha, color, ymin, ymax, label="", offset_bar_color='#040404'):
        ax.fill_between([xmin, xmax], ymin, ymax, color=color, alpha=alpha)
        ax.plot([xmin, xmax], [ymax + 1, ymax + 1], color=offset_bar_color, linewidth=2, label=label)

    def plot_stimulus(self, stimON, stimOFF, ymin, ymax, ax, alpha, highlight_color, label, offset_color):
        for i in range(len(stimON)):
            self.plot_pair(xmin=stimON[i], xmax=stimOFF[i], ax=ax, alpha=alpha, color=highlight_color, ymin=ymin, ymax=ymax,
                    label=(label if i == 0 else ""), offset_bar_color=offset_color)

    def create_mouse_plots(self, prop, filename):
        last_index = np.where(self.time > self.CSoff[-1] + 120)[0][0]

        # dFF plot
        dFF_fig = self.plot_dFF_trace(last_index)
        self.save_plot(dFF_fig, 'dFF', filename)
        print('########## Saved dFF ##########')

        # Z-score plot
        zscore_fig, zscore_ax = self.plot_zscore_trace(prop)
        ymin, ymax = min(self.zScore) - 2, max(self.zScore) + 2
        self.plot_stimulus(self.CSon, self.CSoff, ymin, ymax, zscore_ax, alpha=0.3, highlight_color='#FBF887', label="CS", offset_color='#BB1F1F')
        self.plot_stimulus(self.CSoff - 2, self.CSoff, ymin, ymax + 1, zscore_ax, alpha=0, highlight_color=None, label="US", offset_color='#040404')
        zscore_ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 1.2), ncol=2)
        zscore_ax.grid(False)
        self.save_plot(zscore_fig, 'Zscores', filename)
        print('########## Saved Z-score ##########')

    def start_analysis(self, filename, plot_heatmap=True, plot_auc=True, plot_peth=True, plot_dff_and_zscore=True):
        print('\n********** Starting Analysis **********\n')
        
        if plot_auc:
            self.get_and_plot_AUC(filename)
        if plot_peth:
            self.get_and_plot_PETH(filename)
        if plot_heatmap:
            self.plot_heat_map(8, 4, filename, title = "", ylabel="")
        if plot_dff_and_zscore:
            prop = self.setup_plot_style()
            self.create_mouse_plots(prop, filename)

        print('\n********** Analysis Completed **********\n')

def main(block_path, is_train=True, pre_time=1, post_time=60, auc_log_path='auc.txt'):
    """
    Initialize mouse data processing and start analysis.

    Parameters:
    - block_path (str): Path to the data file.
    - is_train (bool): Whether the data is for training or not.
    - pre_time (int): Pre-stimulus time in seconds.
    - post_time (int): Post-stimulus time in seconds.
    """
    mousename = os.path.basename(block_path)
    print(f'\n\n######## Initializing Mouse {mousename} ########')

    # Create Mouse instance and start analysis
    mouse = Mouse(block_path, isTrain=is_train, PRE_TIME=pre_time, POST_TIME=post_time, aucLogPath=auc_log_path)
    mouse.start_analysis(mousename)

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
    # delete the file if it already exists to avoid appending data
    if os.path.exists(output_path):
        os.remove(output_path)

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sheet_name, data in tqdm(data_sheets.items(), desc="Writing to Excel"):
            if data:
                num_cs = max(len(row) - 1 for row in data)
                column_names = ["Mouse Name"] + [f"CS{i+1}" for i in range(num_cs)]
                df = pd.DataFrame(data, columns=column_names)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    logging.info(f"Excel file created successfully at {output_path}")

def process_directory(dir_path, generate_auc_xlsx=True):
    auc_log_path = os.path.join(os.getcwd(), 'auc.txt')
    auc_xlsx_path = os.path.join(os.getcwd(), 'auc.xlsx')

    # delete the file if it already exists to avoid appending data
    if os.path.exists(auc_log_path):
        os.remove(auc_log_path)

    try:
        for root, _, files in os.walk(dir_path):
            root_lower = root.lower()
            
            # Skip processing if "skip", "archive", or "habituation" is in the path
            if any(keyword in root_lower for keyword in ('archive', 'habituation')):
                logging.info(f"Skipping {root} due to keyword match i.e. 'archive', 'habituation'")
                continue
            
            # Check for .tev files and process if found
            if any(file.endswith('.tev') for file in files):
                is_train = 'train' in root_lower
                logging.info(f"Processing {root}. Is Train: {is_train}")
                try:
                    main(root, is_train=is_train, auc_log_path=auc_log_path)
                    logging.info(f"Processed {root}")
                except Exception as e:
                    logging.error(f"Error processing {root}: {e}")
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user. Proceeding to generate AUC xlsx.")
    finally:
        if generate_auc_xlsx:
            data_sheets = read_data(auc_log_path)
            write_to_excel(data_sheets, auc_xlsx_path)
            logging.info(f"AUC xlsx generated at {auc_xlsx_path}")

if __name__ == "__main__":
    dir_path = input("Enter the directory path: ").strip()
    if os.path.isdir(dir_path):
        process_directory(dir_path)
    else:
        print("The provided path is not a valid directory.")