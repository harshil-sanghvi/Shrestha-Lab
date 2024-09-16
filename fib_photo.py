import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
import tdt
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import matplotlib.font_manager as fm
from scipy.stats import sem
import matplotlib.patches as mpatches
import seaborn as sns
import os

class Mouse:
    def __init__(self, file_path, isTrain, PRE_TIME, POST_TIME, signal = "_465A", control = "_405A", isReins = False):
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

        self.BLOCK_PATH = file_path
        self.data = None
        self.time = None
        self.fs = None
        self.isTrain = isTrain
        self.CSon = None
        self.CSoff = None
        self.USon = None
        self.USoff = None
        self.samplingrate = None
        self.signal_lowpass = None
        self.control_lowpass = None
        self.signal_doubleExpFit = None
        self.control_doubleExpFit = None
        self.signal_doubleFittedCurve = None
        self.control_doubleFittedCurve = None
        self.dFF = None
        self.Y_fit_all = None
        self.Y_dF_all = None
        self.zScore = None
        self.dFF_snips = None
        self.PRE_TIME = PRE_TIME
        self.POST_TIME = POST_TIME
        self.peri_time = None
        self.t1 = 20
        self.savgol_zscore = None
        self.isReins = isReins

        self.__load_data()
        self.__calculate_properties(signal, control)

    #Private Methods
    def __load_data(self):
        """
        Load data from the specified file.

        Reads the data from the file specified by BLOCK_PATH attribute
        and extracts relevant information such as signal, control,
        onset and offset times of CS and US stimuli. This method sets the attribute self.data.

        Returns:
        None
        """
        data = tdt.read_block(self.BLOCK_PATH)
        print(data.epocs)

        if self.isTrain:
            t2 = data.epocs.End_.onset[-1]
            self.data = tdt.read_block(self.BLOCK_PATH, t1=self.t1)
            # self.data = tdt.read_block(self.BLOCK_PATH, t1=data.epocs.PrtB.onset[0], t2=data.epocs.CS__.offset[1] + 1860)

        else:
            self.data = tdt.read_block(self.BLOCK_PATH, t1=self.t1)
            # self.data = tdt.read_block(self.BLOCK_PATH, t1=data.epocs.PrtB.onset[0], t2=data.epocs.CS__.offset[1] + 1860)


    def __calculate_properties(self, signal, control):
        """
        Calculate various properties of the loaded data.

        Calculates properties such as sampling rate, time vector,
        onset and offset times of CS and US stimuli, and performs
        low-pass filtering, photobleaching correction (double-fit subtraction), motion correction(linear fit subtraction).

        Returns:
        None
        """

        self.signal = self.data.streams[signal].data
        self.control = self.data.streams[control].data
        print(len(self.signal), len(self.control))
        max_len = min(len(self.signal), len(self.control))
        self.signal = self.signal[:max_len]
        self.control = self.control[:max_len]

        self.fs = self.data.streams[signal].fs
        self.time = np.linspace(1,len(self.signal), len(self.signal))/self.data.streams[signal].fs
        if self.isTrain:
            self.USon = self.data['epocs']['Shck']['onset'] - self.t1
            self.USoff = self.data['epocs']['Shck']['offset'] - self.t1
        if not self.isReins:
            self.CSon = self.data['epocs']['CS__']['onset'] - self.t1
            self.CSoff = self.data['epocs']['CS__']['offset'] - self.t1
        self.samplingrate = self.data.streams[signal].fs
        self.signal_lowpass, self.control_lowpass = self.__lowpass_filter(self.signal, self.control, self.samplingrate)
        (self.signal_doubleExpFit,
         self.control_doubleExpFit,
         self.signal_doubleFittedCurve,
         self.control_doubleFittedCurve) = self.__doubleFitSubtraction(self.signal_lowpass, self.control_lowpass, self.time)

        self.dFF, self.Y_dF_all, self.Y_fit_all = self.__linearFitSubtraction(self.signal_doubleExpFit, self.control_doubleExpFit, self.signal_doubleFittedCurve)

        self.zScore = (self.Y_dF_all - np.mean(self.Y_dF_all))/np.std(self.Y_dF_all)
        self.savgol_zscore = self.__get_savgol_zscore()
        # self.dFF = self.__remove_outliers(self.dFF)
        # self.zScore = self.__remove_outliers(self.zScore)

        self.dFF_snips, self.peri_time = self.__calculate_dFF_snips(self.signal, self.savgol_zscore, self.fs, self.CSon, self.time, self.PRE_TIME, self.POST_TIME)

    def __lowpass_filter(self, signal, control, samplingrate):
        """
        Apply low-pass filter to the signal and control data.

        Applies a Butterworth low-pass filter to the signal and control
        data with the specified sampling rate.

        Parameters:
        - signal (array): Signal data to be filtered.
        - control (array): Control data to be filtered.
        - samplingrate (int): Sampling rate of the data.

        Returns:
        Tuple containing the filtered signal and control data.
        """
        Order = 6
        Cutoff = 6
        b, a = butter(Order // 2, Cutoff / (samplingrate / 2), btype='low')
        signal_data = filtfilt(b, a, signal)
        control_data = filtfilt(b, a, control)

        return signal_data, control_data

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

        Calculates delta F/F snippets for each CS event based on the signal
        data, z-score, sampling frequency, onset and offset times of CS stimuli,
        time vector, and pre-stimulus and post-stimulus durations.

        Parameters:
        - signal (array): Signal data.
        - zscore (array): Z-score calculated from the signal data.
        - fs (int): Sampling frequency of the data.
        - CSoff (array): Offset times of CS stimuli.
        - time (array): Time vector.
        - PRE_TIME (int): Pre-stimulus duration.
        - POST_TIME (int): Post-stimulus duration.

        Returns:
        Tuple containing the delta F/F snippets and corresponding time vector.
        """
        TRANGE = [-PRE_TIME*np.floor(fs), POST_TIME*np.floor(fs)]
        dFF_snips = []
        array_ind = []
        pre_stim = []
        post_stim = []
        for on in CSon:
            if on < PRE_TIME:
                dFF_snips.append(np.zeros(TRANGE[1]-TRANGE[0]))
            else:
                array_ind.append(np.where(time > on)[0][0])
                pre_stim.append(array_ind[-1] + TRANGE[0])
                post_stim.append(array_ind[-1] + TRANGE[1])
                dFF_snips.append(zscore[int(pre_stim[-1]):int(post_stim[-1])])
        mean_dFF_snips = np.mean(dFF_snips, axis=0)
        std_dFF_snips = np.std(mean_dFF_snips, axis=0)
        peri_time = np.linspace(1, len(mean_dFF_snips), len(mean_dFF_snips))/fs - PRE_TIME
        return dFF_snips, peri_time

    def __remove_outliers(self, signal):
        q25, q75 = np.percentile(signal, [25, 75])
        iqr = q75 - q25
        # Define the thresholds for extreme values based on IQR
        mean = np.mean(signal)
        lower_threshold = mean - 10 * iqr
        upper_threshold = mean + 10 * iqr

        # Find indices of extreme values (both peaks and troughs)
        extreme_indices = np.where((signal < lower_threshold) | (signal > upper_threshold))[0]

        # Create a copy of the signal to modify
        signal_corrected = signal.copy()

        # Replace extreme values with the mean of the entire signal
        signal_corrected[extreme_indices] = mean
        return signal_corrected

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

    def get_and_plot_PETH(self, size_x=8, size_y=6):
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
        self.plot_stimulus_box(ax, xmin=28, xmax=30, ymin=ymin, ymax=ymax + (0.02 * diff), alpha=0, color=cs_highlight_color, 
                        label='US', offset_bar_color=offset_us_color)

        # Axis labels and title
        ax.set_xlabel('Seconds')
        ax.set_ylabel(r'$\Delta$F/F')
        ax.set_title('Peri-Event Trial Responses')
        ax.legend()

        # Save and show the plot
        self.save_plot(fig, 'peth', 'peth')
        plt.show()


    def get_and_plot_AUC(self):
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

        # Plotting AUC values
        plt.figure(figsize=(8, 6))
        index = np.arange(1, num_events + 1)
        bar_width = 0.8

        plt.bar(index, auc_values_cs, bar_width, capsize=5, label='CS', color='#BB1F1F')

        # Formatting the plot
        plt.xlabel("CS Event")
        plt.ylabel("AUC Values")
        plt.title("AUC for CS Events")
        plt.xticks(index)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        self.save_plot(plt, 'auc', 'auc')
        plt.show()

        return auc_values_cs, std_cs

    def plot_heat_map(self, size_x, size_y, title, ylabel=""):
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
        # Set x-axis tick labels based on the pre and post time
        xticklabels = np.arange(-self.PRE_TIME, min(self.POST_TIME, 60) + 10, 10)

        # Calculate xticks based on the number of intervals
        xticks = np.linspace(0, len(self.dFF_snips[0]) - 1, len(xticklabels)).astype(int)

        # Reverse the dFF_snips array for plotting
        dFF_snips_reversed = np.flip(self.dFF_snips, axis=0)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(size_x, size_y))

        # Plot the heatmap using Seaborn
        sns.heatmap(dFF_snips_reversed, cmap='jet', ax=ax)

        # Set labels and title
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Set x-axis ticks and labels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # Set y-axis ticks and labels (reversed and centered)
        yticklabels_reversed = np.arange(1, len(self.dFF_snips) + 1)[::-1]
        ax.set_yticks(np.arange(len(self.dFF_snips)) + 0.5)
        ax.set_yticklabels(yticklabels_reversed)

        # Set plot limits
        ax.set_xlim(0, len(self.dFF_snips[0]))
        ax.set_ylim(0, len(self.dFF_snips))

        # Save and display the plot
        self.save_plot(fig, 'heatmaps', 'heatmaps')
        plt.show()

    def setup_plot_style(self):
        plt.style.use('seaborn-v0_8-white')
        prop = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family="Arial")))
        return prop

    def save_plot(self, fig, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(f'{output_dir}/{filename}.png')

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

    def create_mouse_plots(self, prop):
        last_index = np.where(self.time > self.CSoff[-1] + 120)[0][0]

        # dFF plot
        dFF_fig = self.plot_dFF_trace(last_index)
        self.save_plot(dFF_fig, 'dFF', 'dFF')

        # Z-score plot
        zscore_fig, zscore_ax = self.plot_zscore_trace(prop)
        ymin, ymax = min(self.zScore) - 2, max(self.zScore) + 2
        self.plot_stimulus(self.CSon, self.CSoff, ymin, ymax, zscore_ax, alpha=0.3, highlight_color='#FBF887', label="CS", offset_color='#BB1F1F')
        self.plot_stimulus(self.CSoff - 2, self.CSoff, ymin, ymax + 1, zscore_ax, alpha=0, highlight_color=None, label="US", offset_color='#040404')
        zscore_ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 1.2), ncol=2)
        zscore_ax.grid(False)
        self.save_plot(zscore_fig, 'zscore', 'zscore')

if __name__ == "__main__":
    BLOCK_PATH = '/Users/harshil/Library/CloudStorage/GoogleDrive-Harshil.Sanghvi@stonybrook.edu/Shared drives/NBB_ShresthaLab_SharedDrive/2 Data/Behavior & FibPho/504 SL - Behavior Room/FibPho Data/PL_CAG.GCaMP6f_FCtag.O4E PTC/20230706 PL_CAG.GCaMP6f_FCtag.O4E CT1 PTC/20230710 PL_CAG.GCaMP6f_FCtag.O4E CT1 PTC Training/Pavlovian_cTC_v1-230710-112619/A848-230710-142846'
    mouse = Mouse(BLOCK_PATH, isTrain=True, PRE_TIME=1, POST_TIME=60)
    
    prop = mouse.setup_plot_style()
    mouse.get_and_plot_AUC()
    mouse.get_and_plot_PETH()
    mouse.plot_heat_map(8, 4, title = "", ylabel="")
    mouse.create_mouse_plots(prop)