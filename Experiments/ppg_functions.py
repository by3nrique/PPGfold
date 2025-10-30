import matplotlib.pyplot as plt
from math import floor
import numpy as np
import random
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from scipy.signal import resample, butter, filtfilt
from tqdm import tqdm
from plotly.subplots import make_subplots
import matplotlib.patches as mpatches
from scipy.signal import firwin, filtfilt


def plot_hidden_representation_interactive(Htr, Hts, Ytr, Yts, title, labels,p=[1,1], save=None):
    colorscale = 'Viridis'  # Define colorscale
    # Map your labels to colors
    color_dict = {label: i for i, label in enumerate(labels)}
    
    # Percentage of the data to plot
    p_tr, p_ts = p

    # Randomly select a subset of the data to plot
    idx_tr = np.random.choice(Htr.shape[0], int(p_tr * Htr.shape[0]), replace=False)
    idx_ts = np.random.choice(Hts.shape[0], int(p_ts * Hts.shape[0]), replace=False)

    # Select the subset of data to plot
    Htr = Htr[idx_tr]
    Ytr = Ytr[idx_tr]
    Hts = Hts[idx_ts]
    Yts = Yts[idx_ts]
    
    if Hts.shape[1] == 3:
        # Initialize figures with subplots for 3D scatter plots
        fig = go.Figure()
        
        # Loop through each label to add to the plot for Training set
        for i, label in enumerate(labels):
            idx = Ytr == i  # Find indices of this category
            fig.add_trace(go.Scatter3d(
                x=Htr[idx, 0],
                y=Htr[idx, 1],
                z=Htr[idx, 2],
                mode='markers',
                marker=dict(size=2, color=color_dict[label], colorscale=colorscale, opacity=0.8),
                name=label  # Label for the legend
            ))
            
        # Add general title (suptitle)
        fig.update_layout(title=f'{title} - Training Set')
        # Move legend to top left
        fig.update_layout(legend=dict(x=0, y=1, traceorder='normal', font=dict(size=12)))
        # Show the Training set plot
        fig.show()
        if save is not None:
            fig.write_html(save + title.replace(" ", "") + 'train.html')

        # Repeat for Test set
        fig = go.Figure()
        for i, label in enumerate(labels):
            idx = Yts == i  # Find indices of this category
            fig.add_trace(go.Scatter3d(
                x=Hts[idx, 0],
                y=Hts[idx, 1],
                z=Hts[idx, 2],
                mode='markers',
                marker=dict(size=2, color=color_dict[label], colorscale=colorscale, opacity=0.8),
                name=label  # Label for the legend
            ))
            
        # Add general title (suptitle)
        fig.update_layout(title=f'{title} - Test Set')
        # Move legend to top left
        fig.update_layout(legend=dict(x=0, y=1, traceorder='normal', font=dict(size=12)))
        # Show the Test set plot
        fig.show()
        if save is not None:
            fig.write_html(save + title.replace(" ", "") + 'test.html')

    else:
        # Similar approach for 2D scatter plots
        fig = go.Figure()
        for i, label in enumerate(labels):
            idx = Ytr == i
            fig.add_trace(go.Scatter(
                x=Htr[idx, 0],
                y=Htr[idx, 1],
                mode='markers',
                marker=dict(size=2, color=color_dict[label], colorscale=colorscale, showscale=True),
                name=label
            ))
        fig.update_layout(title=f'{title} - Training Set')
        # Move legend to top left
        fig.update_layout(legend=dict(x=0, y=1, traceorder='normal', font=dict(size=12)))
        fig.show()
        if save is not None:
            fig.write_html(save + title.replace(" ", "") + 'train.html')

        fig = go.Figure()
        for i, label in enumerate(labels):
            idx = Yts == i
            fig.add_trace(go.Scatter(
                x=Hts[idx, 0],
                y=Hts[idx, 1],
                mode='markers',
                marker=dict(size=2, color=color_dict[label], colorscale=colorscale, showscale=True),
                name=label
            ))
        fig.update_layout(title=f'{title} - Test Set')
        # Move legend to top left
        fig.update_layout(legend=dict(x=0, y=1, traceorder='normal', font=dict(size=12)))
        fig.show()
        if save is not None:
            fig.write_html(save + title.replace(" ", "") + 'test.html')

def buffer_signal(signal, frame_size, overlap):
    # Calcula el paso entre frames consecutivos
    step = frame_size - overlap
    # Número de frames
    num_frames = (len(signal) - overlap) // step
    # Inicializa la matriz de salida
    buffer = np.zeros((num_frames, frame_size))
    
    for i in range(num_frames):
        start = i * step
        end = start + frame_size
        buffer[i, :] = signal[start:end]
    return buffer

def split_train_test(X, Y, test_size=0.2, val_size=0.2, random_state=42):
    X_train, X_ts, Y_train, Y_ts = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, Y_train, test_size=val_size/(1-test_size), random_state=random_state)
    return X_tr, X_val, X_ts, Y_tr, Y_val, Y_ts

def process_data(X, y, windows=85, overlap=24, val=0.025, test=0.95):
    # Initialize processed arrays
    Xprocessed_train = []
    Yprocessed_train = []
    Xprocessed_val = []
    Yprocessed_val = []
    Xprocessed_test = []
    Yprocessed_test = []
    signalsTR = []
    signalsVAL = []
    signalsTS = []
    Itr = []
    Ival = []
    Its = []
    
    # If y is a dict with multiple labels
    if type(y) == dict:
        print('Multiple labels detected for y variable')
        y_values = np.zeros((len(list(y.values())[0]), len(y.keys())))
        for i, value in enumerate(y.values()):
            y_values[:, i] = value
        y = np.array(y_values[:, 0])
    else:
        y_values = y
    
    # Iterate over unique labels in y
    for label in np.unique(y):
        filter_indices = np.where(y == label)[0]
        X_BUFFER = []
        for n in filter_indices:
            X_BUFFER.append(X[n])

        Y_BUFFER = np.array(y_values)[filter_indices]

        # Calculate the number of instances for each partition
        num_instances = len(X_BUFFER)
        num_train = max(1, int(num_instances * (1 - val - test)))
        num_val = max(0, int(num_instances * val)) if val > 0 else 0
        num_test = max(0, int(num_instances * test)) if test > 0 else 0

        # Shuffle data to randomize selection for train, val, and test sets
        indices = np.random.permutation(num_instances)

        # Distribute instances to train, val, and test sets
        for i, signal_index in enumerate(indices):
            signal = buffer_signal(X_BUFFER[signal_index], windows, overlap)[:-2]

            if i < num_train:
                Itr.append(filter_indices[signal_index])
                signalsTR.append(X_BUFFER[signal_index])
                Xprocessed_train.extend(signal)
                for _ in signal:
                    Yprocessed_train.append(Y_BUFFER[signal_index])
            elif num_val > 0 and i < num_train + num_val:
                Ival.append(filter_indices[signal_index])
                signalsVAL.append(X_BUFFER[signal_index])
                Xprocessed_val.extend(signal)
                for _ in signal:
                    Yprocessed_val.append(Y_BUFFER[signal_index])
            elif num_test > 0:
                Its.append(filter_indices[signal_index])
                signalsTS.append(X_BUFFER[signal_index])
                Xprocessed_test.extend(signal)
                for _ in signal:
                    Yprocessed_test.append(Y_BUFFER[signal_index])

    # Convert lists to numpy arrays
    Xtr = np.array(Xprocessed_train)
    Ytr = np.array(Yprocessed_train).astype(int)
    Xval = np.array(Xprocessed_val)
    Yval = np.array(Yprocessed_val).astype(int)
    Xts = np.array(Xprocessed_test)
    Yts = np.array(Yprocessed_test).astype(int)

    # Print counts of classes in each dataset
    print('Train labels:', np.unique(Ytr, return_counts=True))
    if val > 0:
        print('Val labels:', np.unique(Yval, return_counts=True))
    else:
        print('No validation set created (val=0).')
    if test > 0:
        print('Test labels:', np.unique(Yts, return_counts=True))
    else:
        print('No test set created (test=0).')

    return Xtr, Ytr, Xval, Yval, Xts, Yts, signalsTR, signalsVAL, signalsTS, Itr, Ival, Its
def resample_signal(signal, original_fs, target_fs):
    """
    Resample the signal to the target frequency.
    """
    if original_fs == target_fs:
        return signal
    
    if original_fs % target_fs == 0:
        # Downsampling by an integer factor
        factor = original_fs // target_fs
        resampled_signal = signal[::factor]
    else:
        # Resampling with an antialiasing lowpass filter
        num_samples = int(len(signal) * target_fs / original_fs)
        resampled_signal = resample(signal, num_samples)
    
    return resampled_signal

def bandpass_filter(signal, fs, lowcut=0.5, highcut=8.0, order=4):
    """
    Apply a Butterworth bandpass filter to the signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band',analog=False)
    filtered_signal  = filtfilt(b, a, signal)
    residual = signal - filtered_signal
    return filtered_signal, residual

def maf_filter(signal, sampling_rate, cutoff_freq=9):
    """
    Apply a moving average-like filter with a precise 9 Hz cut-off frequency.
    In the case of very noisy signals, some high-frequency content can remain in the band-pass filter signal. For this purpose, a 50 ms standard flat (boxcar or top-hat) MAF with a 9 Hz cut-off frequency was applied after the band-pass filtering.
    Parameters:
        signal (numpy.ndarray): Input signal.
        sampling_rate (int): Sampling rate of the signal in Hz.
        cutoff_freq (float): Desired cut-off frequency in Hz (default: 9 Hz).
    
    Returns:
        numpy.ndarray: Filtered signal.
    """
    # Design a FIR filter with a flat (boxcar-like) response
    num_taps = int(0.05 * sampling_rate)  # Approximate 50 ms window duration
    fir_coefficients = firwin(num_taps, cutoff=cutoff_freq, fs=sampling_rate, pass_zero='lowpass')
    
    # Apply the filter using filtfilt for zero-phase delay
    filtered_signal = filtfilt(fir_coefficients, [1], signal)
    
    return filtered_signal

def process_ppg_signal(ppg_signal, original_fs,desired_fs,ff,nfft, apply_moving_average=False):
    # Resample the signal to 100 Hz
    resampled_signal = resample_signal(ppg_signal, original_fs, target_fs=desired_fs)

    if nfft is None:
        spectrum_signal = np.abs(np.fft.fftshift(np.fft.fft(ppg_signal)))
    else:
        spectrum_signal = np.abs(np.fft.fftshift(np.fft.fft(ppg_signal, n=nfft)))

    filtered_signal,residual = bandpass_filter(resampled_signal, fs=desired_fs,lowcut=ff[0], highcut=ff[1])

    # Apply moving average-like filter
    if apply_moving_average:
        filtered_signal = maf_filter(filtered_signal, sampling_rate=desired_fs, cutoff_freq=9)
    # Apply normalization 
    filtered_signal = (filtered_signal - np.mean(filtered_signal))/np.std(filtered_signal)
    
    return filtered_signal, spectrum_signal, residual

def prepare_data(X, y, fs, desired_fs,ff, nfft, apply_moving_average=False):
    X_filtered = []
    spectrums = []
    residuals = []

    for i in tqdm(range(len(X))):
        filtered_signal, spectrum_signal, residual = process_ppg_signal(X[i],fs,desired_fs,ff,nfft, apply_moving_average=False)
        X_filtered.append(filtered_signal)
        spectrums.append(spectrum_signal)
        residuals.append(residual)

    try:    
        X = np.array(X_filtered)
    except ValueError:
        X = X_filtered
    try:
        spectrums = np.array(spectrums)
    except ValueError:
        spectrums = spectrums
    try:
        residuals = np.array(residuals)
    except ValueError:
        residuals = residuals
    return X, spectrums, residuals
