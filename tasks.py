import os
import wfdb
import wget
import zipfile
import pywt
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.signal import butter, lfilter, iirnotch, find_peaks
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from hyperparams import *
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, accuracy_score
import pandas as pd
import json
import neurokit2 as nk
import matplotlib.pyplot as plt



def download_data():
    # 데이터셋 다운로드 및 설치
    wfdb.dl_database('mitdb', dl_dir='mitdb')
    wget.download(pwave_url, "p_wave.zip")
    with zipfile.ZipFile("p_wave.zip", 'r') as zip_ref:
        zip_ref.extractall("mitdb_p_wave")
    os.remove("p_wave.zip")

def get_records(mitdb_dir, p_wave_dir):
    mitdb_records = set(wfdb.get_record_list(mitdb_dir))
    p_wave_records = set([f.replace('.pwave', '') for f in os.listdir(p_wave_dir) if f.endswith('.pwave')])
    return mitdb_records, p_wave_records

def load_ECG_signal(record, dir=mitdb_dir, channels=[0]):
    mitdb_path = os.path.join(dir, record)
    return wfdb.rdsamp(mitdb_path, channels=channels)

def load_ECG_annotations(record, dir, extension):
    mitdb_path = os.path.join(dir, record)
    if not os.path.exists(mitdb_path + '.' + extension):
        return None
    return wfdb.rdann(mitdb_path, extension)

def bandpass_filter(data, lowcut=0.5, highcut=50, fs=360, order=5):
    # TODO : banpass lowcut, highcut 나중에 gridsearch 에 넣을지 고민해보기
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    b_notch, a_notch = iirnotch(60, 30, fs)
    y = lfilter(b_notch, a_notch, y)
    baseline = lfilter([1], [1, 0.995], y)
    return y - baseline


# R 피크 검출 함수 - pantompkins
def get_rpeaks_pantompkins(signal, fs=360):
    def derivative_filter(data):
        # 5점 미분 필터
        return np.convolve(data, [2, 1, 0, -1, -2], mode='same') / 8

    def squaring(data):
        return data ** 2

    def moving_window_integration(data, window_size=int(0.150 * fs)):
        # 150ms 윈도우 적분
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='same')

    def adaptive_threshold(signal, factor_p=0.3, factor_n=0.1, window=150):
        sig_max = np.zeros_like(signal)
        sig_min = np.zeros_like(signal)

        for i in range(len(signal)):
            start = max(0, i - window)
            end = min(len(signal), i + window)
            window_slice = signal[start:end]
            sig_max[i] = np.max(window_slice)
            sig_min[i] = np.min(window_slice)

        threshold_p = sig_min + factor_p * (sig_max - sig_min)
        threshold_n = sig_min + factor_n * (sig_max - sig_min)

        return threshold_p, threshold_n

    # 1. 미분
    differentiated = derivative_filter(signal)

    # 2. 제곱
    squared = squaring(differentiated)

    # 3. 이동 평균 적분
    integrated = moving_window_integration(squared)

    # 4. R피크 검출을 위한 적응형 임계값 설정
    threshold_p, threshold_n = adaptive_threshold(integrated)

    # 5. 피크 검출
    peaks, _ = find_peaks(integrated,
                         height=threshold_n,
                         distance=int(0.2 * fs))  # 최소 R-R 간격 200ms

    # 6. 피크 재검증 및 실제 R피크 위치 조정
    verified_peaks = []
    for peak in peaks:
        if integrated[peak] > threshold_p[peak]:
            # 실제 R피크 위치 찾기: 원본 신호에서 로컬 최대값 찾기
            window_start = max(0, peak - int(0.1 * fs))
            window_end = min(len(signal), peak + int(0.1 * fs))
            actual_peak = window_start + np.argmax(signal[window_start:window_end])
            verified_peaks.append(actual_peak)

    return np.array(sorted(list(set(verified_peaks))))  # 중복 제거 및 정렬


# 개선된 R-피크 검출 함수
def get_rpeaks(signal, level=4, wavelet='sym4', minHR=0.65, fs=360):    
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    
    d4 = coeffs[1]  # d3 is the third last coefficient
    d3 = coeffs[2]  # d4 is the fourth last coefficient

    reconst_signal = pywt.upcoef('d', d3, wavelet, level=3, take=len(signal)) + \
                           pywt.upcoef('d', d4, wavelet, level=4, take=len(signal))

    distance = int(minHR * fs)  
    peaks, _ = find_peaks(reconst_signal, distance=distance)




    # # 후처리1: 진폭 기반 필터링
    # r_peak_amplitudes = signal[r_peaks]
    # amplitude_threshold = np.mean(r_peak_amplitudes) * peakthresh  # 평균 진폭의 peakthresh%를 임계값으로 설정
    # filtered_r_peaks = r_peaks[r_peak_amplitudes > amplitude_threshold]

    # # 후처리2: 너무 가까운 피크 제거
    # min_peak_distance = int(minpeakterm * fs)  # 최소 피크 간 거리 (초)
    # final_r_peaks = []
    # for i, peak in enumerate(filtered_r_peaks):
    #     if i == 0 or peak - final_r_peaks[-1] >= min_peak_distance:
    #         final_r_peaks.append(peak)

    return np.array(peaks)




# 심박분절함수(R-R segmentation & 300 sample resampling)
def segment_heartbeats(signal, rpeaks, target_length=300):
    segments = []

    # 첫 번째와 마지막 R-피크는 제외
    for i in range(1, len(rpeaks)-1):
        # 현재 R-R 간격 계산
        prev_rr = rpeaks[i] - rpeaks[i-1]
        next_rr = rpeaks[i+1] - rpeaks[i]

        # 세그먼트 시작과 끝 지점 설정
        start = rpeaks[i] - int(0.6 * prev_rr)
        end = rpeaks[i] + int(0.6 * next_rr)

        # 신호 범위 체크
        start = max(0, start)
        end = min(len(signal), end)

        # 세그먼트 추출
        segment = signal[start:end]

        # 리샘플링 수행
        if len(segment) >= 2:
            x = np.linspace(0, 1, len(segment))
            x_new = np.linspace(0, 1, target_length)
            f = interp1d(x, segment, kind='linear')
            resampled_segment = f(x_new)
            segments.append(resampled_segment)
        else:
            segments.append(np.zeros(target_length))

    return np.array(segments)

def segmentation(signal, rpeaks, resample_len=300):
    ## TODO: resample_len hyperparameter로 설정할 수 있도록 변경
    ## TODO: resampling code 검토 
    segments = []
    for i in range(len(rpeaks)-1):
        start = rpeaks[i]
        end = rpeaks[i+1]
        segment = signal[start:end]

        # length resampling
        x = np.linspace(0, 1, len(segment))
        x_new = np.linspace(0, 1, resample_len)
        f = interp1d(x, segment, kind='linear')
        resampled_segment = f(x_new)
        segments.append(resampled_segment)

    return np.array(segments)

def IQR_clipping(data):
    res = data.copy()
    for col in range(data.shape[1]):
        col_data = data[:, col]
        non_nan = col_data[~np.isnan(col_data)]
        if non_nan.size > 0:
            q1, q3 = np.percentile(non_nan, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            res[:, col] = np.clip(col_data, lower, upper)
    return res

def fill_nan(data, method='mean'):
    if method not in ['mean', 'median']:
        raise ValueError("Method must be either 'mean' or 'median'.")

    res = data.copy()
    for col in range(data.shape[1]):
        col_data = data[:, col]
        if method == 'mean':
            replacement = np.nanmean(col_data)
        elif method == 'median':
            replacement = np.nanmedian(col_data)
        nan_idx = np.isnan(col_data)
        res[nan_idx, col] = replacement
    return res

def remove_const_features(data):
    selector = VarianceThreshold(threshold=0)
    return selector.fit_transform(data)

def get_ppeaks_manual(signal, r_peaks, wavelet='db6', dynamin=4, dynamax=6, fs=360):
    level = min(max(dynamin, int(np.log2(len(signal))) - 4), dynamax) # P파는 더 낮은 주파수이므로 레벨을 조정
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # P파에 해당하는 세부 계수 선택 (일반적으로 -3 레벨)
    cD = coeffs[-3]

    # P파 강조
    cD = np.abs(cD)

    # 이동 평균 필터 적용
    window_size = int(0.1 * fs)  # 100ms 윈도우
    cD_filtered = np.convolve(cD, np.ones(window_size)/window_size, mode='same')

    # 임계값 설정
    threshold = np.mean(cD_filtered) + 0.5 * np.std(cD_filtered)

    ppeaks = []
    for r_peak in r_peaks:
        start = max(0, r_peak - int(0.3 * fs))  # P파 검색 구간 확장
        end = r_peak
        search_window = cD_filtered[start:end]

        # P파 후보 찾기
        p_candidates = find_peaks(search_window, height=threshold, distance=int(0.2*fs))[0]

        if len(p_candidates) > 0:
            ppeak = start + p_candidates[-1]  # R-피크에 가장 가까운 P파 선택
            ppeaks.append(ppeak)

    return np.array(ppeaks)

def get_ppeaks(record, signal, rpeaks):
    # P-peak 주석 로드 또는 검출
    annotations = load_ECG_annotations(record, pwave_dir, 'pwave')
    if annotations is not None:
        ppeaks = np.array(annotations.sample)
    else:
        ppeaks = get_ppeaks_manual(signal, rpeaks)
        # print(f"P-peak annotations not found for record {record}. P-peaks detected using custom method.")
    return ppeaks

# P-wave와 R-peak 매칭 (첫/끝 R-피크 제외)
def match_pr(rpeaks, ppeaks):
    res_ppeaks = []
    for rpeak in rpeaks:
        # R-peak 이전의 가장 가까운 P-wave 찾기
        previous_p_waves = ppeaks[ppeaks < rpeak]
        if len(previous_p_waves) > 0:
            res_ppeaks.append(previous_p_waves[-1])
        else:
            res_ppeaks.append(None)  
    return np.array(res_ppeaks)


###########################################################################################
# feature extraction functions
###########################################################################################

def safe_divide(a, b, default=0):
    return np.divide(a, b, out=np.full_like(a, default, dtype=float), where=b!=0)

def safe_mean(a, axis=None, default=0):
    return np.nanmean(a, axis=axis) if np.size(a) > 0 else default

def safe_std(a, axis=None, default=0):
    return np.nanstd(a, axis=axis) if np.size(a) > 1 else default

def safe_max(arr):
    return np.max(arr) if isinstance(arr, np.ndarray) and arr.size > 0 else 0

def safe_min(arr):
    return np.min(arr) if isinstance(arr, np.ndarray) and arr.size > 0 else 0


def calculate_rr_std(r_peaks, i, fs, window_size=5):
    start = max(0, i - window_size // 2)
    end = min(len(r_peaks), i + window_size // 2 + 1)
    rr_intervals = np.diff(r_peaks[start:end]) / fs
    return np.std(rr_intervals) if len(rr_intervals) > 1 else 0

def extract_frequency_features(ecg_signal, fs=360):
    fft_result = np.fft.fft(ecg_signal)
    frequencies = np.fft.fftfreq(len(ecg_signal), 1/fs)

    vlf_power = np.sum(np.abs(fft_result[(frequencies >= 0) & (frequencies < 0.04)]) ** 2)
    lf_power = np.sum(np.abs(fft_result[(frequencies >= 0.04) & (frequencies < 0.15)]) ** 2)
    hf_power = np.sum(np.abs(fft_result[(frequencies >= 0.15) & (frequencies < 0.4)]) ** 2)

    lf_hf_ratio = lf_power / hf_power if hf_power != 0 else 0

    return [vlf_power, lf_power, hf_power, lf_hf_ratio]

def calculate_lf_hf_ratio(ecg_signal, fs=360):
    _, _, _, lf_hf_ratio = extract_frequency_features(ecg_signal, fs)
    return lf_hf_ratio


def calc_slope(signal, peak, fs, dx=0.04):
    segment, _, _ = get_segment(signal, peak, 0, dx, fs)
    time = np.arange(len(segment)) / fs
    slope , intercept = np.polyfit(time, segment, 1)
    return slope

def calc_rr_interval(r_peaks, i, fs):
    if i == 0:
        return safe_divide(r_peaks[1] - r_peaks[0], fs)
    elif i == len(r_peaks) - 1:
        return safe_divide(r_peaks[-1] - r_peaks[-2], fs)
    else:
        prev_rr = safe_divide(r_peaks[i] - r_peaks[i-1], fs)
        next_rr = safe_divide(r_peaks[i+1] - r_peaks[i], fs)
        return np.mean([prev_rr, next_rr])
    
def calc_rr_var(r_peaks, current_index, window_size=5, fs=360):
    start = max(0, current_index - window_size)
    end = min(len(r_peaks), current_index + window_size + 1)
    rr_intervals = np.diff(r_peaks[start:end]) / fs  # 샘플링 레이트를 360Hz로 가정
    return safe_std(rr_intervals)

# P파 지속 시간 추정 함수
def estimate_p_duration(ecg_signal, p_peak, fs=360):
    window = int(0.05 * fs)
    start = max(0, p_peak - window)
    end = min(len(ecg_signal), p_peak + window)
    p_segment = ecg_signal[start:end]
    if len(p_segment) == 0:
        return 0
    threshold = 0.1 * max(p_segment)
    above_threshold = np.where(p_segment > threshold)[0]
    return (above_threshold[-1] - above_threshold[0]) / fs if len(above_threshold) > 1 else 0

def calculate_qrs_morphology(qrs_segment):
    if len(qrs_segment) < 2:
        return 0
    q_wave = safe_min(qrs_segment[:len(qrs_segment)//2])
    s_wave = safe_min(qrs_segment[len(qrs_segment)//2:])
    r_wave = safe_max(qrs_segment)
    return safe_divide(r_wave - q_wave, r_wave - s_wave)

# P파 대칭성 계산 함수
def calc_symmetry(p_segment):
    if len(p_segment) < 2:
        return 0
    mid = len(p_segment) // 2
    return np.corrcoef(p_segment[:mid], p_segment[mid:][::-1])[0, 1]

def extract_features(signal, rpeaks, ppeaks, tpeaks, fs=360):
    try:
        feat_scale_list = []  # 동적으로 특징을 저장할 리스트
        feat_not_scale_list = []  # 동적으로 특징을 저장할 리스트
        
        for i, (rpeak, ppeak, tpeak) in enumerate(zip(rpeaks, ppeaks, tpeaks)):
            # QRS & R
            qrs_segment, qrs_start, qrs_end = get_segment(signal, rpeak, 0.04, 0.05, fs)
            qrs_area = calc_wave_area(qrs_segment, fs)
            qrs_duration = safe_divide(qrs_end - qrs_start, fs)
            qrs_amplitude = safe_max(qrs_segment) - safe_min(qrs_segment)
            rr_interval = calc_rr_interval(rpeaks, i, fs)
            rr_var = calc_rr_var(rpeaks, i)

            # T
            t_segment = []
            tpeak_amplitude = 0
            t_wave_area = 0
            qt_interval = 0
            t_inv = 0
            t_slope = 0
            if tpeak is not None:
                t_segment, _, _ = get_segment(signal, tpeak, 0.05, 0.15, fs)
                tpeak_amplitude = signal[tpeak]
                t_wave_area = calc_wave_area(t_segment, fs)
                t_inv = calc_inversion(t_segment)
                t_slope = calc_slope(signal, tpeak, fs)
                qt_interval = safe_divide((tpeak - qrs_start), fs)

            # P
            p_segment = []
            ppeak_amplitude = 0
            p_wave_area = 0
            p_duration = 0
            pr_interval = 0
            p_sym = 0
            if ppeak is not None:
                p_segment, p_end, p_start = get_segment(signal, ppeak, 0.06, 0.06, fs)
                ppeak_amplitude = signal[ppeak] if 0 <= ppeak < len(signal) else 0
                p_wave_area = calc_wave_area(p_segment, fs)
                p_duration = safe_divide(p_end - p_start, fs)
                pr_interval = safe_divide(rpeak - ppeak, fs)
                p_sym = calc_symmetry(p_segment)

            # 기본 특징 설정 (P파 무관)
            feat_scale_list.append([
                # qrs_duration,  # QRS_duration
                # qrs_amplitude,  # QRS_amplitude
                rr_interval,  # RR_interval
                # pr_interval,  # PR_interval 
                # qt_interval,  # QT_interval
                # tpeak_amplitude,  # T_amplitude
                # ppeak_amplitude,  # P_amplitude 
                rr_var,  # RR_variability
                # qrs_area,  # QRS_area
                # p_duration,  # P_duration 
                # t_wave_area,  # T_area
                # p_wave_area,  # P_wave_area
            ])

            feat_not_scale_list.append([
                # t_inv,  # T_inversion
                # p_sym,  # P_symmetry 
                # t_slope,  # T_slope 
            ])

        # 리스트를 numpy 배열로 변환
        feat_scale = np.array(feat_scale_list)
        feat_not_scale = np.array(feat_not_scale_list)

        return feat_scale, feat_not_scale

    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        return np.zeros((len(rpeaks), 15)), np.zeros((len(rpeaks), 3))  # 에러 시 zero array 반환

    
def get_segment(signal, peak, time_before, time_after, fs):
    start = max(0, peak - int(time_before * fs))
    end = min(len(signal), peak + int(time_after * fs))
    segment = signal[start:end]
    return segment, start, end

################################################################################################

def feature_scaling(data, feature_range=(-1, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def find_closest_idx(arr, target):
    # target이 삽입될 인덱스를 찾습니다.
    idx = np.searchsorted(arr, target)
    
    # 배열의 시작 혹은 끝에 해당하는 경우 처리
    if idx == 0:
        return 0
    if idx == len(arr):
        return len(arr) - 1
    
    # 삽입 위치(idx)와 그 이전 위치(idx-1)의 값을 비교하여 target과 더 가까운 쪽을 선택
    if abs(arr[idx] - target) < abs(target - arr[idx - 1]):
        return idx
    else:
        return idx - 1
    
def find_closest_value(data:dict, target):
    # 각 key와 target의 차이가 가장 작은 key 선택
    closest_key = min(data.keys(), key=lambda k: abs(k - target))
    return data[closest_key]
    
def extract_labels(rpeaks, symbols):
    labels = []
    for rpeak in rpeaks:
        label = find_closest_value(symbols, rpeak)
        labels.append(label)
    return labels
    
def group_labels(label):
    if label in ['N','L','R','e','j']:
        return 'N' # N (Normal)
    elif label in ['A','a','J','S']:  
        return 'S' # S (Supraventricular ectopic)
    elif label in ['V','E']:  
        return 'V' # V (Ventricular ectopic)
    elif label in ['F']:  
        return 'F' # F (Fusion)
    elif label in ['/','f']:
        return 'Q' # Q (Unknown/Paced)
    else:
        return 'O' # other

def print_label_distribution(y):
    # 레이블 분포 계산
    labels, counts = np.unique(y, return_counts=True)

    # 백분율 계산
    percentages = (counts / counts.sum()) * 100

    # 레이블과 퍼센트 출력
    for label, percentage in zip(labels, percentages):
        print(f"Label: {label}, Percentage: {percentage:.2f}%")


def one_hot_encoder(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # label 값 정수로 변환
    y_onehot = to_categorical(y_encoded)  # One-hot encoding 수행

    # class_name classification report에 matching하기위해 저장
    class_names = le.classes_

    return y_onehot, class_names

def calc_metrics(y_test, y_pred, labels):
    def calc_specificity(y_test, y_pred, labels):
        conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(labels)))
        specificity = []

        for i in range(len(labels)):
            tn = np.sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1))  # TN: 행/열 삭제로 계산
            fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]  # FP: 해당 열의 합에서 TP를 제외
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        return specificity
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=range(len(labels)))
    specificity = calc_specificity(y_test, y_pred, labels)

    return precision, recall, f1, specificity

def calc_global_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return {'accuracy': accuracy}


def get_metric_df(precision, recall, f1, specificity, class_names):
    # 결과를 DataFrame으로 정리
    data = {
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity,
    }
    metrics_df = pd.DataFrame(data)
    return metrics_df

def df_to_csv_colab(df, filename):
    from google.colab import files
    df.to_csv(filename)
    files.download(filename)

def dict_to_json_colab(data, filename):
    from google.colab import files
    # Write the dictionary to a .json file
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    # Download the file
    files.download(filename)


def print_label_dist(y):
    unique_labels, counts = np.unique(y, return_counts=True)
    total_count = len(y)

    # 결과 출력 (백분율로 변환)
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_count) * 100
        print(f"{label}: {count} occurrences ({percentage:.2f}%)")

def check_data_split_no_overlap(train, val, test):
    # 각 리스트를 집합으로 변환
    set1 = set(train)
    set2 = set(val)
    set3 = set(test)
    
    # 교집합 검사
    if set1 & set2 or set1 & set3 or set2 & set3:
        return False  # 겹치는 원소가 있음
    return True  # 겹치는 원소가 없음


def calc_wave_area(segment, fs):
    from scipy.integrate import simpson
    """
    주어진 신호 세그먼트의 넓이를 계산.

    Parameters:
    - segment: 신호 세그먼트 (numpy 배열)
    - fs: 샘플링 주파수

    Returns:
    - wave_area: 세그먼트 넓이 (적분 값)
    """
    if len(segment) == 0:
        return 0
    time = np.arange(len(segment)) / fs
    wave_area = simpson(np.abs(segment), x=time)
    return wave_area



def get_tpeaks(signal, r_peaks, wavelet='sym4', dynamin=3, dynamax=5, fs=360, search_window=(0.1, 0.25)):
    """
    T-peak를 검출하는 함수.

    Parameters:
    - signal: ECG 신호
    - r_peaks: R-peak의 인덱스 리스트
    - wavelet: 사용될 웨이블릿 (기본값: 'sym4')
    - dynamin: 웨이블릿 변환 최소 레벨
    - dynamax: 웨이블릿 변환 최대 레벨
    - fs: 샘플링 주파수 (Hz)
    - search_window: R-peak 이후 T-wave 탐색 윈도우 (초 단위, 기본값: (0.2, 0.6))

    Returns:
    - tpeaks: 검출된 T-peak 인덱스 배열
    """
    # 1. T-wave가 포함된 신호 성분만 강조하기 위해 Wavelet Transform 사용
    level = min(max(dynamin, int(np.log2(len(signal))) - 2), dynamax) # -4->3으로 변경. (-2로 수정해볼만 함.)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cd = coeffs[-2]  # T-wave를 포함할 가능성이 높은 세부 성분 선택 -2
    squared = cd ** 2

    # 2. R-peak 기반으로 T-wave 탐색
    tpeaks = []
    for r_peak in r_peaks:
        # R-peak 이후 탐색 윈도우 정의
        start = r_peak + int(search_window[0] * fs)
        end = r_peak + int(search_window[1] * fs)
        if end > len(signal):  # 경계를 넘어가는 경우 방지
            end = len(signal)
        if start >= len(signal):
            break

        # 탐색 윈도우 내에서 최대값의 인덱스를 T-peak로 간주
        window = squared[start:end]
        if len(window) > 0:
            local_max_index = np.argmax(window)
            t_peak_index = start + local_max_index
            tpeaks.append(t_peak_index)

    return np.array(tpeaks)

# P-wave와 R-peak 매칭 (첫/끝 R-피크 제외)- 수정
def match_tr(rpeaks, tpeaks):
    res_tpeaks = []
    for rpeak in rpeaks:
        # R-peak 이후의 가장 가까운 P-wave 찾기
        previous_t_waves = tpeaks[rpeak < tpeaks]
        if len(previous_t_waves) > 0:
            res_tpeaks.append(previous_t_waves[0])
        else:
            # # P-wave가 없는 경우, R-peak에서 일정 거리 뺀 값 사용
            # res_ppeaks.append(max(0, rpeak - int(0.2 * 360)))  # 0.2초를 가정, 샘플링 레이트 360Hz
            # P-wave가 없는 경우, None
            res_tpeaks.append(None)
    return np.array(res_tpeaks, dtype=object)  # None을 포함할 수 있도록 dtype=object 사용


def calc_inversion(t_segment):
    return 1 if safe_min(t_segment) < 0 else 0



# 웨이블릿 변환 및 최적 스케일 찾기
def find_optimal_scale(ecg_signal, scales, wavelet, threshold_method="mean"):
    """
    모든 스케일에 대해 웨이블릿 변환을 수행하고 최적 스케일을 찾는 함수
    
    Args:
        ecg_signal (numpy array): 입력 ECG 신호
        scales (list): 웨이블릿 스케일의 리스트 (a 값)
        wavelet : 웨이블릿 (예: 'db3')
        threshold_method (str): 임계값 계산 방법 ("mean" 또는 "percentile")
    
    Returns:
        optimal_scale (int): 최적의 스케일
        maxima_per_scale (dict): 각 스케일의 국부적 최대값 정보
    """
    # local max
    maxima_per_scale = {}
    global_max_value = -np.inf
    optimal_scale = None
    
    # 모든 스케일에 대해 웨이블릿 변환 수행
    for scale in scales:
        # 웨이블릿 변환 수행
        coeffs, _ = pywt.cwt(ecg_signal, [scale], wavelet)
        coeffs = coeffs[0]  # 웨이블릿 계수 (첫 번째 결과)

        # 국부적 최대값 찾기
        maxima_indices = [
            i for i in range(1, len(coeffs) - 1)
            if coeffs[i] > coeffs[i - 1] and coeffs[i] > coeffs[i + 1]
        ]
        maxima_values = coeffs[maxima_indices]
        
        # 임계값 계산
        if threshold_method == "mean":
            threshold = np.mean(maxima_values)
        elif threshold_method == "percentile":
            threshold = np.percentile(maxima_values, 75)  # 상위 25%만 선택
        else:
            raise ValueError("Invalid threshold method")
        
        # 임계값을 초과하는 국부적 최대값만 필터링
        valid_maxima = [v for v in maxima_values if v > threshold]
        
        # 스케일별 최대값 저장
        maxima_per_scale[scale] = {
            "maxima_indices": maxima_indices,
            "maxima_values": valid_maxima,
            "threshold": threshold
        }
        
        # 최적 스케일 찾기
        if len(valid_maxima) > 0 and max(valid_maxima) > global_max_value:
            global_max_value = max(valid_maxima)
            optimal_scale = scale

    return optimal_scale, maxima_per_scale


def create_wavelet_instance(ex):
    base_wavelet = type(ex)
    class DiscreteContinuousWaveletEx(base_wavelet):
        def __init__(self, name='', filter_bank=None):
            base_wavelet.__init__(name, filter_bank)  
            self.complex_cwt = False # wavelet 계수  True: 복소수, False: 실수
    return DiscreteContinuousWaveletEx(ex.name)


def adjust_rpeaks(sig, rpeaks, height_threshold=0.4):
        """
        Adjust R-peak positions based on detected peaks within the segments between R-peaks.

        Returns:
        tuple: A tuple containing adjusted R-peak positions and a list of candidate R-peaks.
        """
        if len(rpeaks) == 0:
            return [], []
        
        adjusted_rpeaks = rpeaks.copy()
        candidate_rpeaks = []

        # Calculate the height threshold for peak detection
        peak_height = (np.max(sig) - np.min(sig)) * height_threshold

        for i in range(len(rpeaks) - 1):
            # Extract the segment between two R-peaks
            segment = sig[rpeaks[i]:rpeaks[i + 1]]
            rpeak_left = sig[rpeaks[i]]
            rpeak_right = sig[rpeaks[i + 1]]

            try:
                # Detect peaks within the segment using the height threshold
                peaks, _ = find_peaks(segment, height=peak_height)
            except Exception as e:
                print(f"Error finding peaks: {e}")
                continue

            if len(peaks) == 1:
                abs_peak_idx = rpeaks[i] + peaks[0]
                left_diff = segment[peaks[0]] - rpeak_left
                right_diff = segment[peaks[0]] - rpeak_right

                if left_diff >= 0 and right_diff >= 0:
                    # Replace the closer R-peak with the detected peak
                    if (abs_peak_idx - rpeaks[i]) > (rpeaks[i + 1] - abs_peak_idx):
                        adjusted_rpeaks[i + 1] = abs_peak_idx
                    else:
                        adjusted_rpeaks[i] = abs_peak_idx

                elif left_diff * right_diff <= 0:
                    # Replace the lower R-peak with the detected peak
                    if rpeak_left < rpeak_right:
                        adjusted_rpeaks[i] = abs_peak_idx
                    else:
                        adjusted_rpeaks[i + 1] = abs_peak_idx

            elif len(peaks) == 2:
                abs_peak_idx_1 = rpeaks[i] + peaks[0]
                abs_peak_idx_2 = rpeaks[i] + peaks[1]

                diff_1 = segment[peaks[0]] - rpeak_left
                diff_2 = segment[peaks[1]] - rpeak_right

                if diff_1 > 0 and diff_2 > 0:
                    # Replace R-peaks with the detected peaks
                    adjusted_rpeaks[i] = abs_peak_idx_1
                    adjusted_rpeaks[i + 1] = abs_peak_idx_2

        # Check the difference between original and adjusted R-peaks
        diff = np.abs(rpeaks - adjusted_rpeaks)
        diff = diff[diff > 0]

        if len(diff) >= len(rpeaks) * 0.5:
            for i in range(len(adjusted_rpeaks) - 1):
                segment = sig[adjusted_rpeaks[i]:adjusted_rpeaks[i + 1]]
                try:
                    peaks, _ = find_peaks(segment)
                    max_peak_idx = np.argmax(segment[peaks])
                    max_peaks = adjusted_rpeaks[i] + peaks[max_peak_idx]
                    candidate_rpeaks.append(max_peaks)
                except Exception as e:
                    print(f"Error processing candidate peaks: {e}")
                    continue

            # Remove duplicates from candidate R-peaks
            candidate_rpeaks = [peak for peak in candidate_rpeaks if peak not in adjusted_rpeaks]
            return adjusted_rpeaks, candidate_rpeaks

        return adjusted_rpeaks, []

def get_rpeaks(sig, ecg_clean_method, ecg_peaks_method, fs=360):
    cleaned_sig = nk.ecg_clean(sig, sampling_rate=fs, method=ecg_clean_method)
    try:
        _, peaks = nk.ecg_peaks(cleaned_sig, sampling_rate=fs, method=ecg_peaks_method, correct_artifacts=True)
        rpeaks = peaks['ECG_R_Peaks']
    except:
        rpeaks = []
    return rpeaks

## plotting
def ecg_peak_plot(sig, rpeaks, adj_rpeaks):
    sample_indices = np.arange(len(sig))
    # 플롯 생성
    plt.figure(figsize=(12, 6))

    # 원본 ECG 신호 플롯
    plt.subplot(2, 1, 1)
    plt.plot(sample_indices, sig, label="ECG Signal", color="black")
    plt.scatter(rpeaks, sig[rpeaks], color="red", label="Local Maxima", marker="o")  # ✅ 국부 최대값 표시

    plt.subplot(2, 1, 2)
    plt.plot(sample_indices, sig, label="ECG Signal", color="black")
    plt.scatter(adj_rpeaks, sig[adj_rpeaks], color="red", label="Local Maxima", marker="o")  # ✅ 국부 최대값 표시


    plt.title("ECG Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

def ecg_adjpeak_label_plotting(sig, adj_rpeaks, labels):
    fig, ax = plt.subplots(figsize=(20, 4))
    sample_indices = np.arange(len(sig))

    # 기본 ECG 파형 그리기
    ax.plot(sample_indices, sig, label="ECG Signal", color="black")
    # R-peak 지점 표시
    ax.scatter(adj_rpeaks, sig[adj_rpeaks], color="red", label="Local Maxima", marker="o")

    # R-peak마다 라벨 달기
    for rp, label in zip(adj_rpeaks, labels):
        ax.text(rp, -0.05, label, 
                transform=ax.get_xaxis_transform(),  # x는 데이터 좌표, y는 축 좌표
                ha='center', 
                va='top', 
                color='blue')

    ax.set_title("ECG Signal with Peak Labels")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.legend()

    # 아래쪽 여백이 잘리지 않도록 y범위 조정
    ax.set_ylim(min(sig)*1.1, max(sig)*1.1)

    plt.tight_layout()
    plt.show()