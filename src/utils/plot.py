import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def label_record_hist(label, records):
    # 예시 데이터 생성
    label_dist_df = pd.DataFrame({'label': label, 'record': records})

    # label과 record별 count 집계
    df_count = label_dist_df.groupby(['label', 'record']).size().unstack(fill_value=0)

    # stacked bar plot
    ax = df_count.plot(
        kind='bar', 
        stacked=True, 
        figsize=(12, 8),   # 그래프 크기
        width=0.8         # 막대 너비 조절
    )

    # 범례 설정
    plt.legend(
        title='Record',       # 범례 제목
        bbox_to_anchor=(1.05, 1),  # 그래프 오른쪽 바깥에 위치하도록 설정
        loc='upper left', 
        borderaxespad=0,
        ncol=2               # 범례를 2열로 표시(필요에 따라 조정)
    )

    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Label별 Record 분포")

    # 레이아웃 자동 조정(그래프 요소 겹침 방지)
    plt.tight_layout()
    plt.show()

def label_hist(label):
    label_counts = pd.Series(label).value_counts()
    label_counts.plot(kind='bar')
    plt.show()
    print(label_counts)