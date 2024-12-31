import os
import wfdb
import wget
import zipfile
import pywt
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.signal import butter, lfilter, iirnotch, find_peaks
from scipy.interpolate import interp1d
from hyperparams import *

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



# 개선된 R-피크 검출 함수
def get_rpeaks(signal, peakthresh=0.6, minpeakterm=0.2, wavelet='db4', dynamin=3, dynamax=5, fs=360):    
    level = min(max(dynamin, int(np.log2(len(signal))) - 4), dynamax)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    cd = coeffs[-2]
    squared = cd ** 2
    window_size = int(0.1 * fs)
    convolved = np.convolve(squared, np.ones(window_size), 'same') / window_size

    # 동적 임계값 설정
    threshold = np.mean(convolved) + 2 * np.std(convolved)

    # R-피크 후보 검출
    r_peaks, _ = find_peaks(signal, distance=int(0.2*fs), height=threshold)

    # 후처리1: 진폭 기반 필터링
    r_peak_amplitudes = signal[r_peaks]
    amplitude_threshold = np.mean(r_peak_amplitudes) * peakthresh  # 평균 진폭의 peakthresh%를 임계값으로 설정
    filtered_r_peaks = r_peaks[r_peak_amplitudes > amplitude_threshold]

    # 후처리2: 너무 가까운 피크 제거
    min_peak_distance = int(minpeakterm * fs)  # 최소 피크 간 거리 (초)
    final_r_peaks = []
    for i, peak in enumerate(filtered_r_peaks):
        if i == 0 or peak - final_r_peaks[-1] >= min_peak_distance:
            final_r_peaks.append(peak)

    return np.array(final_r_peaks)




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

def get_ppeaks_manual(signal, r_peaks, wavelet='db4', dynamin=4, dynamax=6, fs=360):
    wavelet = 'db4'
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
        else:
            # P파를 찾지 못한 경우, R-피크 이전의 고정된 지점을 P파로 가정
            ppeaks.append(max(0, r_peak - int(0.2 * fs)))

    return np.array(ppeaks)

def get_ppeaks(record, signal, rpeaks):
    # P-peak 주석 로드 또는 검출
    annotations = load_ECG_annotations(record, pwave_dir, 'pwave')
    if annotations is not None:
        ppeaks = np.array(annotations.sample)
    else:
        ppeaks = get_ppeaks_manual(signal, rpeaks)
        print(f"P-peak annotations not found for record {record}. P-peaks detected using custom method.")
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
            # P-wave가 없는 경우, R-peak에서 일정 거리 뺀 값 사용
            res_ppeaks.append(max(0, rpeak - int(0.2 * 360)))  # 0.2초를 가정, 샘플링 레이트 360Hz
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

def calculate_t_slope(t_segment, fs):
    if len(t_segment) < 2:
        return 0
    time = np.arange(len(t_segment)) / fs
    slope, _ = np.polyfit(time, t_segment, 1)
    return slope

def calculate_rr_interval(r_peaks, i, fs):
    if i == 0:
        return safe_divide(r_peaks[1] - r_peaks[0], fs)
    elif i == len(r_peaks) - 1:
        return safe_divide(r_peaks[-1] - r_peaks[-2], fs)
    else:
        prev_rr = safe_divide(r_peaks[i] - r_peaks[i-1], fs)
        next_rr = safe_divide(r_peaks[i+1] - r_peaks[i], fs)
        return np.mean([prev_rr, next_rr])
    
def calculate_rr_variability(r_peaks, current_index, window_size=5):
    start = max(0, current_index - window_size)
    end = min(len(r_peaks), current_index + window_size + 1)
    rr_intervals = np.diff(r_peaks[start:end]) / 360  # 샘플링 레이트를 360Hz로 가정
    return safe_std(rr_intervals)

# P파 지속 시간 추정 함수
def estimate_p_duration(ecg_signal, p_wave, fs=360):
    window = int(0.05 * fs)
    start = max(0, p_wave - window)
    end = min(len(ecg_signal), p_wave + window)
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
def calculate_p_symmetry(ecg_signal, p_wave, fs=360):
    window = int(0.05 * fs)
    start = max(0, p_wave - window)
    end = min(len(ecg_signal), p_wave + window)
    p_segment = ecg_signal[start:end]
    if len(p_segment) < 2:
        return 0
    mid = len(p_segment) // 2
    return np.corrcoef(p_segment[:mid], p_segment[mid:][::-1])[0, 1]

def extract_features(signal, rpeaks, ppeaks, fs=360):
    try:
        features = np.zeros((len(rpeaks), 16))  # 특징 수를 16개로 줄임
        for i, (rpeak, ppeak) in enumerate(zip(rpeaks, ppeaks)):
            qrs_start = max(0, rpeak - int(0.1 * fs))
            qrs_end = min(len(signal), rpeak + int(0.1 * fs))
            qrs_segment = signal[qrs_start:qrs_end]

            t_start = min(len(signal), rpeak + int(0.05 * fs))
            t_end = min(len(signal), rpeak + int(0.4 * fs))
            t_segment = signal[t_start:t_end]

            features[i] = [
                safe_divide(qrs_end - qrs_start, fs),  # QRS_duration
                safe_max(qrs_segment) - safe_min(qrs_segment),  # QRS_amplitude
                calculate_rr_interval(rpeaks, i, fs),  # RR_interval
                safe_divide(rpeak - ppeak, fs),  # PR_interval
                safe_divide(t_end - qrs_start, fs),  # QT_interval
                safe_max(t_segment) - safe_min(t_segment),  # T_amplitude
                signal[ppeak] if 0 <= ppeak < len(signal) else 0,  # P_amplitude
                calculate_rr_variability(rpeaks, i),  # RR_variability
                1 if safe_min(t_segment) < 0 else 0,  # T_inversion
                np.sum(np.abs(qrs_segment)),  # QRS_area
                estimate_p_duration(signal, ppeak, fs),  # P_duration
                np.sum(np.abs(t_segment)),  # T_area
                calculate_rr_std(rpeaks, i, fs),  # RR_std
                calculate_qrs_morphology(qrs_segment),  # QRS_morphology
                calculate_t_slope(t_segment, fs),  # T_slope
                calculate_p_symmetry(signal, ppeak, fs),  # P_symmetry
            ]

        return features
    except Exception as e:
        print(f"Error in extract_all_features: {str(e)}")
        return np.array([])