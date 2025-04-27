from config import MITDB_PATH, FS, EX_LABELS,HRV_WINDOW
from src.preprocess.data_loader import get_mitdb_records, load_ECG_signal, load_symbols
from src.preprocess.signal_process import ecg_clean, get_rpeaks, adjust_rpeaks, segmentation, compute_segment_hrv
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from src.preprocess.label_process import extract_labels, group_labels
import numpy as np
def run_pipeline(model_type, hrv_window=HRV_WINDOW):
    all_segments = []
    all_labels = []
    all_records = []
    all_hrv = []
    i=0
    mitdb = get_mitdb_records()
    for record in tqdm(mitdb):
        print(record)
        # load ECG signal & annotations
        sig = load_ECG_signal(record)
        dct_symbols = load_symbols(record, MITDB_PATH, extension='atr', EX_LABELS=EX_LABELS) 

        # sig cleaning
        sig_cleaned = ecg_clean(sig, FS)
        # rpeak detection
        rpeaks = get_rpeaks(sig_cleaned, ecg_peaks_method='neurokit')
        adj_rpeaks, candid_rpeaks = adjust_rpeaks(sig_cleaned, rpeaks)

        # sig normalization
        scaler = StandardScaler()
        sig_scaled = scaler.fit_transform(sig_cleaned.reshape(-1, 1)).flatten()
        if model_type == 0:
            # segmetation based on rpeaks
            segments = segmentation(sig_scaled, adj_rpeaks)

            # label extraction & grouping
            labels = extract_labels(adj_rpeaks, dct_symbols)
            labels = list(map(group_labels, labels))
            labels = labels[1:]  # segmentation을 하기 때문에 마지막은 제거

        elif model_type == 1: 
            # feature extraction (HRV)
            hrv = compute_segment_hrv(adj_rpeaks, sampling_rate=FS, hrv_window=hrv_window)
            all_hrv.append(hrv)

            # segmetation based on rpeaks
            segments = segmentation(sig_scaled, adj_rpeaks)
            segments = segments[:-hrv_window+1]  # 마지막 min_beats 개는 제거 (HRV와 개수 맞추기)

            # label extraction & grouping
            labels = extract_labels(adj_rpeaks, dct_symbols)
            labels = list(map(group_labels, labels))
            labels = labels[1:]  # segmentation을 하기 때문에 마지막은 제거
            labels = labels[:-hrv_window+1]  # 마지막 min_beats 개는 제거 (HRV와 개수 맞추기)

        # split을 위한 record 인덱스 array 생성
        record_idx = np.array([record]*len(labels)) 
        # 데이터를 리스트에 추가
        all_labels.append(labels)
        all_records.append(record_idx)
        all_segments.append(segments)   
        # i+=1
        # if i == 1:
        #     break

        x1 = np.concatenate(all_segments, axis=0)
        if model_type == 0:
            x2 = None
        elif model_type == 1:
            x2 = np.concatenate(all_hrv, axis=0)
        y = np.concatenate(all_labels, axis=0)
        records = np.concatenate(all_records, axis=0)
        
    return x1, x2, y, records   