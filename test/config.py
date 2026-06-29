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
    'seed': 39,
    'fs' : 360,
    'hrv_window' : 5,
    'ex_labels' : ['+', '[', ']', '!', 'Q', 'x', '"', '|', '~',],
    'model_type' : 0, # 0: x1 only, 1: x1,x2
    'model_arch': 'cnn',  # 'cnn' or 'cnn_attn'
    'resample_len' : 300, # ecg resample length
    
    # Augmentation settings
    'n_aug_dict': {'N': 1, 'S': 4, 'V': 1, 'Q': 1},  # S 클래스 4배 증강으로 증가
    'noise_level_dict': {'N': 0.01, 'S': 0.02, 'V': 0.01, 'Q': 0.01},  # 노이즈 레벨 유지
    
    # Advanced augmentation parameters
    'aug_type': 'all',  # 'time_warp', 'magnitude_warp', 'crop_pad', 'spec_aug', 'all'
    'time_warp_sigma': 0.2,
    'magnitude_warp_sigma': 0.2,
    'magnitude_warp_knot': 4,
    'crop_ratio': 0.1,
    'freq_mask': 0.1,
    'time_mask': 0.1,


    # Loss & Weight parameters
    # Loss function 설정
    'focal_loss_gamma': 0,    # focal loss gamma (0: BCE 사용, >0: focal loss 사용)
    'focal_loss_alpha': 0.25,   # focal loss alpha (focal loss의 positive class 가중치)

    # Class weight 설정
    'use_class_weight': True,  # class weight 사용 여부
    'class_weight_multiply': 2.5,  # S 클래스 가중치 기본 배수
    'class_weight_max': 6.0,  # S 클래스 가중치 최대값
}

def _dict_to_str(d, prefix):
    # 예: {'N': 1, 'S': 4, 'V': 2, 'Q': 1} -> 'nN1S4V2Q1'
    return prefix + ''.join([f"{k}{v}" for k, v in d.items()])

def _dict_to_str_float(d, prefix):
    # 예: {'N': 0.01, 'S': 0.01, 'V': 0.01, 'Q': 0.01} -> 'nlN001S001V001Q001'
    return prefix + ''.join([f"{k}{str(v).replace('.', '').zfill(3)}" for k, v in d.items()])

def _get_aug_type_str(aug_type, **kwargs):
    """Convert augmentation type and parameters to string"""
    if aug_type == 'all':
        return 'all'
    elif aug_type == 'time_warp':
        return f'tw{str(kwargs.get("sigma", 0.2)).replace(".", "")}'
    elif aug_type == 'magnitude_warp':
        return f'mw{str(kwargs.get("sigma", 0.2)).replace(".", "")}k{kwargs.get("knot", 4)}'
    elif aug_type == 'crop_pad':
        return f'cp{str(kwargs.get("crop_ratio", 0.1)).replace(".", "")}'
    elif aug_type == 'spec_aug':
        return f'sa{str(kwargs.get("freq_mask", 0.1)).replace(".", "")}t{str(kwargs.get("time_mask", 0.1)).replace(".", "")}'
    return aug_type

def _get_loss_str():
    """Loss function과 class weight 설정을 문자열로 변환"""
    parts = []
    if HYPERPARAMS['focal_loss_gamma'] > 0:
        parts.append(f"focal{str(HYPERPARAMS['focal_loss_gamma']).replace('.', '')}")
    if HYPERPARAMS['use_class_weight']:
        parts.append(f"cw{HYPERPARAMS['class_weight_multiply']}")
    return '_'.join(parts) if parts else 'bce'  # 아무것도 없으면 'bce' 반환

EXPERIMENT_NAME = (
    f"Sbinary_augmentADV"
    f"_{_get_loss_str()}"  # Loss function 설정 표시
    f"_{_dict_to_str(HYPERPARAMS['n_aug_dict'], 'n')}"
    f"_{_dict_to_str_float(HYPERPARAMS['noise_level_dict'], 'nl')}"
    f"_{_get_aug_type_str(HYPERPARAMS['aug_type'], **{k: v for k, v in HYPERPARAMS.items() if k.startswith(('time_warp', 'magnitude_warp', 'crop_ratio', 'freq_mask', 'time_mask'))})}"
    f"_seed{HYPERPARAMS['seed']}"
    f"_model{HYPERPARAMS['model_type']}"
    f"_{HYPERPARAMS.get('model_arch', 'cnn')}"  # 모델 아키텍처 구분 (cnn or cnn_attn)
)





