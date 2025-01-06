import matplotlib.pyplot as plt
import numpy as np

def show_rppeaks(ecg_signal, r_peaks, p_waves, fs=360, dpi=600):
    plt.figure(figsize=(20, 10),  dpi=dpi)
    time = np.arange(len(ecg_signal)) / fs
    plt.plot(time, ecg_signal)
    plt.plot(r_peaks / fs, ecg_signal[r_peaks], 'ro', label='R-peaks')
    plt.plot(p_waves / fs, ecg_signal[p_waves], 'go', label='P-peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with R-peaks and P-peaks')
    plt.legend()
    plt.show()

def plot_metric_hist(precision, recall, specificity, class_names, width=0.2):
    x = np.arange(len(class_names))
    plt.bar(x - width, precision, width=width, label="Precision")
    plt.bar(x, recall, width=width, label="Recall")
    plt.bar(x + width, specificity, width=width, label="Specificity")
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.title("Class-wise Performance Metrics")
    plt.xticks(x, class_names)
    plt.legend()
    plt.show()