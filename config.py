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
    'n_aug': 4,           # S 클래스 증강 배수 (deprecated, 개별 dict 사용 권장)
    'noise_level': 0.01,  # S 클래스 증강 노이즈 세기 (deprecated, 개별 dict 사용 권장)
    'n_aug_dict': {'N': 1, 'S': 4, 'V': 2, 'Q': 1},  # 각 클래스별 증강 배수
    'noise_level_dict': {'N': 0.01, 'S': 0.01, 'V': 0.01, 'Q': 0.01},  # 각 클래스별 노이즈 세기
    
    # Advanced augmentation parameters
    'augmentation_config': {
        'S': {  # S class specific augmentation
            'gaussian_noise': {'noise_level': 0.01},
            'time_warp': {'sigma': 0.2},
            'amplitude_scale': {'sigma': 0.1},
            'random_crop': {'crop_ratio': 0.8},
            'mixup': {'alpha': 0.2}
        },
        'V': {  # V class specific augmentation
            'gaussian_noise': {'noise_level': 0.01},
            'time_warp': {'sigma': 0.15}
        },
        'Q': {  # Q class specific augmentation
            'gaussian_noise': {'noise_level': 0.01},
            'amplitude_scale': {'sigma': 0.1}
        }
    }
}

def _dict_to_str(d, prefix):
    # 예: {'N': 1, 'S': 4, 'V': 2, 'Q': 1} -> 'nN1S4V2Q1'
    return prefix + ''.join([f"{k}{v}" for k, v in d.items()])

def _dict_to_str_float(d, prefix):
    # 예: {'N': 0.01, 'S': 0.01, 'V': 0.01, 'Q': 0.01} -> 'nlN001S001V001Q001'
    return prefix + ''.join([f"{k}{str(v).replace('.', '').zfill(3)}" for k, v in d.items()])

EXPERIMENT_NAME = (
    f"augmentALL_"
    f"{_dict_to_str(HYPERPARAMS['n_aug_dict'], 'n')}"
    f"_{_dict_to_str_float(HYPERPARAMS['noise_level_dict'], 'nl')}"
    f"_seed{HYPERPARAMS['seed']}_model{HYPERPARAMS['model_type']}"
)




# EXPERIMENT_NAME 예시: f"augmentS_n{HYPERPARAMS['n_aug']}_nl{str(HYPERPARAMS['noise_level']).replace('.', '')}_seed{HYPERPARAMS['seed']}_model{HYPERPARAMS['model_type']}"

