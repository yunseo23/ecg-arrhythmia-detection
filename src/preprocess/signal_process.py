import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def ecg_clean(sig, fs, method='biosppy'):
    return nk.ecg_clean(sig, sampling_rate=fs, method=method)

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

def get_rpeaks(sig, ecg_peaks_method, fs=360):
    try:
        _, peaks = nk.ecg_peaks(sig, sampling_rate=fs, method=ecg_peaks_method, correct_artifacts=True)
        rpeaks = peaks['ECG_R_Peaks']
    except:
        rpeaks = []
    return rpeaks

def compute_segment_hrv(rpeaks, sampling_rate, hrv_window):
    """
    rpeaks 리스트를 이용해 세그먼트별 시간 영역 HRV 지표를 계산하는 함수.
    각 세그먼트는 연속된 R-peak 그룹으로 정의하며,
    최소 min_beats 이상의 R-peak가 있어야 HRV 계산을 수행.
    """
    segment_hrv_list = []
    
    # R-peak가 2개 미만이면 세그먼트를 만들 수 없으므로 빈 DataFrame 반환
    if len(rpeaks) < 2:
        return pd.DataFrame()
    
    # 각 세그먼트: 인접한 R-peak들을 그룹화 (세그먼트 개수는 len(rpeaks)-1)
    for i in range(len(rpeaks)-1):
        # 여기서는 i번째부터 i+min_beats번째까지를 한 세그먼트로 정의.
        segment_rpeaks = rpeaks[i:i+hrv_window+1]
        
        # 세그먼트에 충분한 R-peak가 있는 경우에만 HRV 계산 수행
        if len(segment_rpeaks) >= hrv_window+1:
            # nk.hrv()는 기본적으로 시간, 주파수, 비선형 등 모든 영역의 지표를 계산하려고 함.
            # 짧은 세그먼트에서는 주파수 영역 계산 시 에러가 발생할 수 있으므로,
            # 시간 영역 지표만 계산하는 nk.hrv_time()을 사용합니다.
            hrv_segment = nk.hrv_time(segment_rpeaks, sampling_rate=sampling_rate).to_numpy().flatten()
            
            # 세그먼트 식별을 위한 인덱스 추가
            # hrv_segment["Segment"] = i
            
            # 계산된 HRV 결과 DataFrame을 리스트에 추가
            segment_hrv_list.append(hrv_segment)
            
    # 모든 세그먼트의 HRV 결과를 하나의 DataFrame으로 결합
    if segment_hrv_list:
        # return pd.concat(segment_hrv_list, ignore_index=True)
        return segment_hrv_list
    else:
        # return pd.DataFrame()
        return []