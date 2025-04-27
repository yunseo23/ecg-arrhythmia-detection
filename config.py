# 데이터 경로
MITDB_PATH = 'mitdb'
pwave_path = './mitdb_p_wave/mit-bih-arrhythmia-database-p-wave-annotations-1.0.0'
pwave_url = "https://physionet.org/static/published-projects/pwave/mit-bih-arrhythmia-database-p-wave-annotations-1.0.0.zip"


SEED = 20 # random seed
FS = 360 # signal frequency
HRV_WINDOW = 5
EX_LABELS = ['+', '[', ']', '!', 'Q', 'x', '"', '|', '~',] # labels to exclude
MODEL_TYPE = 0 # 0: x1 only, 1: x1,x2
RESAMPLE_LEN = 300 # ecg resample length

