# 데이터 경로
MITDB_PATH = 'mitdb'
pwave_path = './mitdb_p_wave/mit-bih-arrhythmia-database-p-wave-annotations-1.0.0'
pwave_url = "https://physionet.org/static/published-projects/pwave/mit-bih-arrhythmia-database-p-wave-annotations-1.0.0.zip"

# random seed
SEED = 20
# signal frequency
FS = 360
#
HRV_WINDOW = 5
# labels to exclude
EX_LABELS = ['+', '[', ']', '!', 'Q', 'x', '"', '|', '~',]
MODEL_TYPE = 0
'''
0: x1 only
1: x1,x2
'''

