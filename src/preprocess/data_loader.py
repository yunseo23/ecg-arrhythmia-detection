import wfdb
import wget
import zipfile
import os
import numpy as np
from config import MITDB_PATH

def download_data(pwave_url):
    # 데이터셋 다운로드 및 설치
    wfdb.dl_database('mitdb', dl_dir='mitdb')
    wget.download(pwave_url, "p_wave.zip")
    with zipfile.ZipFile("p_wave.zip", 'r') as zip_ref:
        zip_ref.extractall("mitdb_p_wave")
    os.remove("p_wave.zip")

def get_mitdb_records(mitdb_path):
    return set(wfdb.get_record_list(mitdb_path))

def get_pwave_records(p_wave_path):
    return set([f.replace('.pwave', '') for f in os.listdir(p_wave_path) if f.endswith('.pwave')])

def load_ECG_signal(record, path=MITDB_PATH, channels=[0]):
    mitdb_path = os.path.join(path, record)
    sig, _ = wfdb.rdsamp(mitdb_path, channels=channels)
    sig = np.squeeze(sig)
    return sig

def load_ECG_annotations(record, path, extension):
    mitdb_path = os.path.join(path, record)
    if not os.path.exists(mitdb_path + '.' + extension):
        return None
    return wfdb.rdann(mitdb_path, extension)

def load_symbols(record, path, extension, EX_LABELS=None):
    ann = load_ECG_annotations(record, path, extension)
    ann_sample = ann.sample # rpeak 근처 annotation
    symbols = ann.symbol  # label
    dct_symbols = dict(zip(ann_sample, symbols)) # key: rpeak idx, value: label
    if EX_LABELS is not None:
        dct_symbols = {k: v for k, v in dct_symbols.items() if v not in EX_LABELS} # filter symbols
    return dct_symbols





