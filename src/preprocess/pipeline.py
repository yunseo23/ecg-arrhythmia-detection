from config import MITDB_PATH, HYPERPARAMS
from src.preprocess.data_loader import get_mitdb_records, load_ECG_signal, load_symbols
from src.preprocess.signal_process import ecg_clean, get_rpeaks, adjust_rpeaks, segmentation, compute_segment_hrv
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from src.preprocess.label_process import extract_labels, group_labels
import numpy as np
import matplotlib.pyplot as plt

def undersample(x1, x2, y, records, target_class='S', random_state=42):
    idx_target = np.where(y == target_class)[0]
    n_target = len(idx_target)
    idx_other = np.where(y != target_class)[0]
    if n_target == 0:
        raise ValueError(f"No samples found for class {target_class}")
    np.random.seed(random_state)
    idx_other_sampled = np.random.choice(idx_other, n_target, replace=False)
    idx_total = np.concatenate([idx_target, idx_other_sampled])
    np.random.shuffle(idx_total)
    x1_new = x1[idx_total]
    y_new = y[idx_total]
    records_new = records[idx_total]
    if x2 is not None:
        x2_new = x2[idx_total]
    else:
        x2_new = None
    return x1_new, x2_new, y_new, records_new

def augment_signal(signal, noise_level=0.01):
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def augment_class(x1, x2, y, records, target_class='S', n_aug=2, noise_level=0.01):
    idx_s = np.where(y == target_class)[0]
    x1_s = x1[idx_s]
    y_s = y[idx_s]
    records_s = records[idx_s]
    if x2 is not None:
        x2_s = x2[idx_s]
    augmented_x1 = []
    augmented_x2 = [] if x2 is not None else None
    augmented_y = []
    augmented_records = []
    for _ in range(n_aug):
        for i, sig in enumerate(x1_s):
            augmented_x1.append(augment_signal(sig, noise_level))
            if x2 is not None:
                augmented_x2.append(x2_s[i])
            augmented_y.append(target_class)
            augmented_records.append(records_s[i])
    x1_aug = np.concatenate([x1, np.array(augmented_x1)], axis=0)
    if x2 is not None:
        x2_aug = np.concatenate([x2, np.array(augmented_x2)], axis=0)
    else:
        x2_aug = None
    y_aug = np.concatenate([y, np.array(augmented_y)], axis=0)
    records_aug = np.concatenate([records, np.array(augmented_records)], axis=0)
    return x1_aug, x2_aug, y_aug, records_aug

def augment_all_classes(x1, x2, y, records, n_aug_dict=None, noise_level_dict=None):
    classes = np.unique(y)
    x1_aug, x2_aug, y_aug, records_aug = x1, x2, y, records
    for cls in classes:
        n_aug = n_aug_dict.get(cls, 0) if n_aug_dict else 0
        noise_level = noise_level_dict.get(cls, 0.01) if noise_level_dict else 0.01
        if n_aug > 0:
            x1_aug, x2_aug, y_aug, records_aug = augment_class(
                x1_aug, x2_aug, y_aug, records_aug,
                target_class=cls, n_aug=n_aug, noise_level=noise_level
            )
    return x1_aug, x2_aug, y_aug, records_aug

def time_warp(signal, sigma=0.2):
    """Apply time warping to the signal"""
    orig_steps = np.arange(signal.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(signal.shape[0]))
    warp_steps = (np.cumsum(random_warps))
    warp_steps = warp_steps * (signal.shape[0]-1)/warp_steps[-1]
    ret_signal = np.interp(orig_steps, warp_steps, signal)
    return ret_signal

def magnitude_warp(signal, sigma=0.2, knot=4):
    """Apply magnitude warping to the signal"""
    orig_steps = np.arange(signal.shape[0])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
    warp_steps = (np.linspace(0, signal.shape[0]-1, num=knot+2))
    ret_signal = np.interp(orig_steps, warp_steps, random_warps)
    return signal * ret_signal

def random_crop_pad(signal, crop_ratio=0.1):
    """Apply random cropping and padding"""
    signal_len = signal.shape[0]
    crop_len = int(signal_len * crop_ratio)
    start_idx = np.random.randint(0, signal_len - crop_len)
    cropped = signal[start_idx:start_idx + crop_len]
    pad_left = np.random.normal(0, 0.1, size=start_idx)
    pad_right = np.random.normal(0, 0.1, size=signal_len - (start_idx + crop_len))
    return np.concatenate([pad_left, cropped, pad_right])

def spec_augment(signal, freq_mask=0.1, time_mask=0.1):
    """Apply SpecAugment-like augmentation"""
    # FFT
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal))
    
    # Frequency masking
    if freq_mask > 0:
        mask_len = int(len(freq) * freq_mask)
        mask_start = np.random.randint(0, len(freq) - mask_len)
        fft[mask_start:mask_start + mask_len] = 0
    
    # Time masking
    if time_mask > 0:
        mask_len = int(len(signal) * time_mask)
        mask_start = np.random.randint(0, len(signal) - mask_len)
        signal[mask_start:mask_start + mask_len] = 0
    
    # Inverse FFT
    return np.real(np.fft.ifft(fft))

def augment_signal_advanced(signal, aug_type='all', **kwargs):
    """Apply advanced augmentation techniques"""
    if aug_type == 'time_warp':
        return time_warp(signal, sigma=kwargs.get('sigma', 0.2))
    elif aug_type == 'magnitude_warp':
        return magnitude_warp(signal, sigma=kwargs.get('sigma', 0.2), knot=kwargs.get('knot', 4))
    elif aug_type == 'crop_pad':
        return random_crop_pad(signal, crop_ratio=kwargs.get('crop_ratio', 0.1))
    elif aug_type == 'spec_aug':
        return spec_augment(signal, freq_mask=kwargs.get('freq_mask', 0.1), time_mask=kwargs.get('time_mask', 0.1))
    elif aug_type == 'all':
        # Apply all augmentations with some probability
        if np.random.random() < 0.3:
            signal = time_warp(signal)
        if np.random.random() < 0.3:
            signal = magnitude_warp(signal)
        if np.random.random() < 0.3:
            signal = random_crop_pad(signal)
        if np.random.random() < 0.3:
            signal = spec_augment(signal)
        return signal
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

def augment_class_advanced(x1, x2, y, records, target_class='S', n_aug=2, aug_type='all', **kwargs):
    """Advanced version of augment_class using new augmentation techniques"""
    idx_s = np.where(y == target_class)[0]
    x1_s = x1[idx_s]
    y_s = y[idx_s]
    records_s = records[idx_s]
    if x2 is not None:
        x2_s = x2[idx_s]
    
    augmented_x1 = []
    augmented_x2 = [] if x2 is not None else None
    augmented_y = []
    augmented_records = []
    
    for _ in range(n_aug):
        for i, sig in enumerate(x1_s):
            augmented_x1.append(augment_signal_advanced(sig, aug_type=aug_type, **kwargs))
            if x2 is not None:
                augmented_x2.append(x2_s[i])
            augmented_y.append(target_class)
            augmented_records.append(records_s[i])
    
    x1_aug = np.concatenate([x1, np.array(augmented_x1)], axis=0)
    if x2 is not None:
        x2_aug = np.concatenate([x2, np.array(augmented_x2)], axis=0)
    else:
        x2_aug = None
    y_aug = np.concatenate([y, np.array(augmented_y)], axis=0)
    records_aug = np.concatenate([records, np.array(augmented_records)], axis=0)
    
    return x1_aug, x2_aug, y_aug, records_aug

def visualize_rpeak_adjustment(sig_cleaned, rpeaks, adj_rpeaks, record, fs=360, time_window=20):
    """
    R-peak 검출과 조정 결과를 시각화하는 함수
    
    Parameters:
    - sig_cleaned: 정제된 ECG 신호
    - rpeaks: 원본 R-peak 위치
    - adj_rpeaks: 조정된 R-peak 위치
    - record: 레코드 번호
    - fs: 샘플링 주파수
    - time_window: 표시할 시간 윈도우 (초)
    """
    # 시간 윈도우에 해당하는 샘플 수
    window_samples = int(time_window * fs)
    
    # 신호의 시작 부분만 표시 (너무 긴 신호는 보기 어려움)
    end_sample = min(window_samples, len(sig_cleaned))
    
    # 시간 축 생성
    time_axis = np.arange(end_sample) / fs
    
    # 윈도우 내의 R-peak들만 필터링
    rpeaks_in_window = rpeaks[rpeaks < end_sample]
    adj_rpeaks_in_window = adj_rpeaks[adj_rpeaks < end_sample]
    
    # 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # 첫 번째 플롯: 원본 R-peaks
    ax1.plot(time_axis, sig_cleaned[:end_sample], 'g-', linewidth=1, label='Initial preprocessed ECG')
    ax1.plot(rpeaks_in_window / fs, sig_cleaned[rpeaks_in_window], 'bo', 
             markersize=8, label=f'Detected R-peaks ({len(rpeaks_in_window)})')
    ax1.set_title(f'Auto-corrected ECG Signal with Detected R-peaks Using NeuroKit2', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 두 번째 플롯: 조정된 R-peaks
    ax2.plot(time_axis, sig_cleaned[:end_sample], 'm-', linewidth=1, label='Initial preprocessed ECG')
    ax2.plot(adj_rpeaks_in_window / fs, sig_cleaned[adj_rpeaks_in_window], 'ro', 
             markersize=8, label=f'Final Adjusted R-peaks ({len(adj_rpeaks_in_window)})')
    ax2.set_title(f'Final Preprocessing Result: ECG Signal with Adjusted R-peaks after Position Optimization', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # R-peak 조정 통계 출력
    original_count = len(rpeaks)
    adjusted_count = len(adj_rpeaks)
    print(f"Record {record}:")
    print(f"  Original R-peaks: {original_count}")
    print(f"  Adjusted R-peaks: {adjusted_count}")
    print(f"  Difference: {adjusted_count - original_count}")
    print(f"  Adjustment rate: {abs(adjusted_count - original_count) / original_count * 100:.2f}%")
    print("-" * 50)
    
def visualize_ecg_cleaning(sig_original, sig_cleaned, record, fs=360, time_window=20):
    """
    원본 ECG 신호와 정제된 ECG 신호를 비교 시각화하는 함수
    
    Parameters:
    - sig_original: 원본 ECG 신호
    - sig_cleaned: 정제된 ECG 신호 (ecg_clean 적용 후)
    - record: 레코드 번호
    - fs: 샘플링 주파수
    - time_window: 표시할 시간 윈도우 (초)
    """
    # 시간 윈도우에 해당하는 샘플 수
    window_samples = int(time_window * fs)
    
    # 신호의 시작 부분만 표시 (너무 긴 신호는 보기 어려움)
    end_sample = min(window_samples, len(sig_original))
    
    # 시간 축 생성
    time_axis = np.arange(end_sample) / fs
    
    # 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # 첫 번째 플롯: 원본 ECG 신호
    ax1.plot(time_axis, sig_original[:end_sample], 'b-', linewidth=1, label='Raw ECG Signal')
    ax1.set_title(f'Raw ECG Signal from MIT-BIH Arrhythmia Database', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 두 번째 플롯: 정제된 ECG 신호
    ax2.plot(time_axis, sig_cleaned[:end_sample], 'r-', linewidth=1, label='Initial preprocessed ECG Signal')
    ax2.set_title(f'ECG Signal after Preprocessing and Noise Removal with NeuroKit2', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 신호 통계 출력
    print(f"Record {record} - Signal Statistics:")
    print(f"  Original signal - Mean: {np.mean(sig_original):.4f}, Std: {np.std(sig_original):.4f}")
    print(f"  Cleaned signal - Mean: {np.mean(sig_cleaned):.4f}, Std: {np.std(sig_cleaned):.4f}")
    print("-" * 60)

def preprocess_pipeline(apply_undersample=False, target_class='S', apply_augment=False, n_aug=None, noise_level=None, apply_augment_all=False, n_aug_dict=None, noise_level_dict=None, visualize_rpeaks=False, visualize_cleaning=False):
    FS = HYPERPARAMS['fs']
    EX_LABELS = HYPERPARAMS['ex_labels']
    HRV_WINDOW = HYPERPARAMS['hrv_window']
    MODEL_TYPE = HYPERPARAMS['model_type']
    if n_aug is None:
        n_aug = HYPERPARAMS.get('n_aug', 2)
    if noise_level is None:
        noise_level = HYPERPARAMS.get('noise_level', 0.01)
    all_segments = []
    all_labels = []
    all_records = []
    all_hrv = []
    i=0
    mitdb = get_mitdb_records()
    for record in tqdm(mitdb):
        print(record)
        # load ECG signal & annotations
        sig = load_ECG_signal(record)  # 원본 신호
        dct_symbols = load_symbols(record, MITDB_PATH, extension='atr', EX_LABELS=EX_LABELS) 

        # sig cleaning
        sig_cleaned = ecg_clean(sig, FS)  # 정제된 신호
        
        # ECG 정제 과정 시각화 (옵션)
        if visualize_cleaning:
            visualize_ecg_cleaning(sig, sig_cleaned, record, FS)
            # 또는 겹쳐서 보고 싶다면:
            # visualize_ecg_cleaning_overlay(sig, sig_cleaned, record, FS)
        
        # rpeak detection
        rpeaks = get_rpeaks(sig_cleaned, ecg_peaks_method='neurokit')
        adj_rpeaks, candid_rpeaks = adjust_rpeaks(sig_cleaned, rpeaks)
        
        # R-peak 시각화 (옵션)
        if visualize_rpeaks:
            visualize_rpeak_adjustment(sig_cleaned, rpeaks, adj_rpeaks, record, FS)

        # sig normalization
        scaler = StandardScaler()
        sig_scaled = scaler.fit_transform(sig_cleaned.reshape(-1, 1)).flatten()
        if MODEL_TYPE == 0:
            # segmetation based on rpeaks
            segments = segmentation(sig_scaled, adj_rpeaks)

            # label extraction & grouping
            labels = extract_labels(adj_rpeaks, dct_symbols)
            labels = list(map(group_labels, labels))
            labels = labels[1:]  # segmentation을 하기 때문에 마지막은 제거

        elif MODEL_TYPE == 1: 
            # feature extraction (HRV)
            hrv = compute_segment_hrv(adj_rpeaks, sampling_rate=FS, hrv_window=HRV_WINDOW)
            all_hrv.append(hrv)

            # segmetation based on rpeaks
            segments = segmentation(sig_scaled, adj_rpeaks)
            segments = segments[:-HRV_WINDOW+1]  # 마지막 min_beats 개는 제거 (HRV와 개수 맞추기)

            # label extraction & grouping
            labels = extract_labels(adj_rpeaks, dct_symbols)
            labels = list(map(group_labels, labels))
            labels = labels[1:]  # segmentation을 하기 때문에 마지막은 제거
            labels = labels[:-HRV_WINDOW+1]  # 마지막 min_beats 개는 제거 (HRV와 개수 맞추기)

        # split을 위한 record 인덱스 array 생성
        record_idx = np.array([record]*len(labels)) 
        # 데이터를 리스트에 추가
        all_labels.append(labels)
        all_records.append(record_idx)
        all_segments.append(segments)   

    x1 = np.concatenate(all_segments, axis=0)
    if MODEL_TYPE == 0:
        x2 = None
    elif MODEL_TYPE == 1:
        x2 = np.concatenate(all_hrv, axis=0)
    y = np.concatenate(all_labels, axis=0)
    records = np.concatenate(all_records, axis=0)

    if apply_augment_all:
        x1, x2, y, records = augment_all_classes(x1, x2, y, records, n_aug_dict=n_aug_dict, noise_level_dict=noise_level_dict)
    elif apply_augment:
        x1, x2, y, records = augment_class_advanced(
            x1, x2, y, records,
            target_class='S',
            n_aug=4,
            aug_type='all'  # 모든 증강 기법을 30% 확률로 적용
        )

    if apply_undersample:
        x1, x2, y, records = undersample(x1, x2, y, records, target_class=target_class)

    return x1, x2, y, records   

def preprocess_pipeline_binary(apply_augment=False, n_aug=1, noise_level=0.01, visualize_rpeaks=False, visualize_cleaning=False):
    """
    이진 분류(S vs Non-S)를 위한 전처리 파이프라인
    """
    # 기존 파이프라인 실행
    x1, x2, y, records = preprocess_pipeline(
        apply_undersample=False,
        target_class='S',
        apply_augment=apply_augment,
        n_aug=n_aug,
        noise_level=noise_level,
        visualize_rpeaks=visualize_rpeaks,
        visualize_cleaning=visualize_cleaning  # 추가된 매개변수
    )
    
    # 라벨을 이진으로 변환 (S=1, others=0)
    y_binary = np.array([1 if label == 'S' else 0 for label in y])
    
    return x1, x2, y_binary, records