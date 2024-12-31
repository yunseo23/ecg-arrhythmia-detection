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