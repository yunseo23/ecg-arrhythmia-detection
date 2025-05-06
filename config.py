# 데이터 경로
MITDB_PATH = 'mitdb'
pwave_path = './mitdb_p_wave/mit-bih-arrhythmia-database-p-wave-annotations-1.0.0'
pwave_url = "https://physionet.org/static/published-projects/pwave/mit-bih-arrhythmia-database-p-wave-annotations-1.0.0.zip"


# path
GRIDSEARCH_PATH = './gridsearch_seed/'
TEST_PATH = './test/'
RESULT_PATH = './res/'  # 실험 결과 저장 경로
DIRS_TO_CREATE = [GRIDSEARCH_PATH, TEST_PATH, RESULT_PATH]

HYPERPARAMS ={
    'seed': 20,
    'fs' : 360,
    'hrv_window' : 5,
    'ex_labels' : ['+', '[', ']', '!', 'Q', 'x', '"', '|', '~',],
    'model_type' : 0, # 0: x1 only, 1: x1,x2
    'resample_len' : 300, # ecg resample length
}

