from src.models.model import CNNx1OnlyModel,CNNModel, BinaryCNNModel, BinaryCNNAttentionModel
from src.preprocess.label_process import one_hot_encoder
from src.evaluation.evaluate import calc_metrics, get_metric_df
from src.utils.plot import plot_metric_hist
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import os
import pandas as pd
from src.utils.utils import export_hyperparams
from config import HYPERPARAMS, RESULT_PATH, EXPERIMENT_NAME

def split_data(x1, y, records, seed, x2=None):
    """
    데이터를 StratifiedGroupKFold를 사용하여 Train/Validation/Test로 분할하는 함수입니다.
    
    Parameters:
        x1 (np.array): 주요 입력 데이터.
        y (np.array): 레이블.
        records (np.array): 그룹 정보 (예: subject id 등).
        seed (int): random_state 설정을 위한 시드 값.
        x2 (np.array, optional): 추가 입력 데이터. 제공되지 않으면 None.
        
    Returns:
        dict: 'train', 'val', 'test' 키를 가지며,
              각 키에 해당하는 값은 {'x1': ..., 'x2': ..., 'y': ...} 형태의 딕셔너리입니다.
              x2가 없는 경우, 'x2'의 값은 None이 됩니다.
    """


    # x2가 제공되면 x1과 x2를 결합, 아니면 x1만 사용
    if x2 is not None:
        n_x1 = x1.shape[1]
        combined = np.hstack((x1, x2))
    else:
        combined = x1

    # 첫 번째 분할: Train+Validation / Test 분할
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_val_idx, test_idx in sgkf.split(combined, y, groups=records):
        break

    train_val_data = combined[train_val_idx]
    y_train_val = y[train_val_idx]
    records_train_val = records[train_val_idx]

    # 두 번째 분할: Train / Validation 분할 (Train+Validation 데이터 내에서)
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in sgkf_val.split(train_val_data, y_train_val, groups=records_train_val):
        break

    if x2 is not None:
        # x2가 제공된 경우: x1, x2를 분리
        x1_train = train_val_data[train_idx][:, :n_x1]
        x2_train = train_val_data[train_idx][:, n_x1:]
        x1_val = train_val_data[val_idx][:, :n_x1]
        x2_val = train_val_data[val_idx][:, n_x1:]
        x1_test = combined[test_idx][:, :n_x1]
        x2_test = combined[test_idx][:, n_x1:]
    else:
        x1_train = train_val_data[train_idx]
        x1_val = train_val_data[val_idx]
        x1_test = combined[test_idx]
        x2_train = None
        x2_val = None
        x2_test = None

    train = {"x1": x1_train, "x2": x2_train, "y": y_train_val[train_idx]}
    val = {"x1": x1_val,   "x2": x2_val,   "y": y_train_val[val_idx]}
    test = {"x1": x1_test,  "x2": x2_test,  "y": y[test_idx]}
    
    
    return train ,val, test



# 클래스 가중치 계산 함수
def compute_class_weights(y_train):
    # class weight 생성
    unique_labels = np.unique(y_train)
    # 'balanced' 모드는 각 클래스가 데이터셋에서 등장하는 비율에 반비례하도록 가중치 계산
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=y_train
    )

    # scikit-learn이 반환한 array를 {클래스: 가중치} 형태의 딕셔너리로 변환
    class_weights = {
        label: weight for label, weight in zip(unique_labels, class_weights_array)
    }
    return class_weights

# One-hot 인코딩 함수 (기존 one_hot_encoder 함수 활용)
def one_hot_encode(y_train, y_val, y_test):
    # one-hot encoding
    y_train_oh, class_names = one_hot_encoder(y_train)
    y_val_oh, _ = one_hot_encoder(y_val)
    y_test_oh, _ = one_hot_encoder(y_test)
    return y_train_oh, y_val_oh, y_test_oh, class_names

def train_and_evaluate(model_type, train, val, test,
                       y_train_oh, y_val_oh,y_test_oh, class_names, class_weights, seed):
    # model_type: 'x1x2' 또는 'x1only'
    if model_type == 1:
        # model initialization
        x1_shape = (train['x1'].shape[1], 1)
        x2_shape = (train['x2'].shape[1],)
        n_classes = y_train_oh.shape[1]
        model = CNNModel(x1_shape, x2_shape, n_classes)   
        # model training
        model.fit([train['x1'],train['x2']], y_train_oh, [val['x1'],val['x2']], y_val_oh, train['y'], class_weight=class_weights)
        # model evaluation
        test_loss, test_accuracy, test_auc = model.evaluate([test['x1'],test['x2']], y_test_oh)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        # prediction
        y_pred = model.predict([test['x1'],test['x2']])
        
    elif model_type == 0:
        # model initialization
        x1_shape = (train['x1'].shape[1], 1)
        n_classes = y_train_oh.shape[1]
        model = CNNx1OnlyModel(x1_shape, n_classes)   
        # model training
        model.fit(train['x1'], y_train_oh, val['x1'], y_val_oh, train['y'], class_weight=class_weights)
        # model evaluation
        test_loss, test_accuracy, test_auc = model.evaluate(test['x1'], y_test_oh)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        # prediction
        y_pred = model.predict(test['x1'])
    else:
        raise ValueError("model_type은 'x1x2' 또는 'x1only'여야 합니다.")
    
    # 평가 지표 계산
    y_test_labels = np.argmax(y_test_oh, axis=1)
    precision, recall, f1, specificity = calc_metrics(y_test_labels, y_pred, class_names)
    print(classification_report(y_test_labels, y_pred, target_names=class_names))
    
    metric_df = get_metric_df(precision, recall, f1, specificity, class_names)
    metric_df['seed'] = seed
    # histogram
    plot_metric_hist(precision, recall, specificity, class_names)
    return model, metric_df

def train_and_evaluate_binary(model_type, train, val, test, y_train, y_val, y_test, class_weights, seed):
    """Train and evaluate binary classification model"""
    if model_type != 0:
        raise ValueError("Binary classification only supports model_type 0 (x1 only)")
    
    # model initialization
    x1_shape = (train['x1'].shape[1], 1)
    
    # Select model based on model_arch parameter
    model_arch = HYPERPARAMS.get('model_arch', 'cnn')
    if model_arch == 'cnn':
        model = BinaryCNNModel(x1_shape)
    elif model_arch == 'cnn_attn':
        model = BinaryCNNAttentionModel(x1_shape)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")
        
    print(f"Using model architecture: {model_arch}")
    
    # model training
    model.fit(train['x1'], y_train, val['x1'], y_val, class_weight=class_weights)
    
    # model evaluation
    test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(test['x1'], y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    # prediction
    y_pred = model.predict(test['x1'])
    
    # 평가 지표 계산
    precision, recall, f1, specificity = calc_metrics(y_test, y_pred, labels=[0, 1])
    
    # 각 클래스별 메트릭 출력
    print("\nClass 0 (Non-S):")
    print(f"Precision: {precision[0]:.4f}")
    print(f"Recall: {recall[0]:.4f}")
    print(f"F1 Score: {f1[0]:.4f}")
    print(f"Specificity: {specificity[0]:.4f}")
    
    print("\nClass 1 (S):")
    print(f"Precision: {precision[1]:.4f}")
    print(f"Recall: {recall[1]:.4f}")
    print(f"F1 Score: {f1[1]:.4f}")
    print(f"Specificity: {specificity[1]:.4f}")
    
    # 결과를 DataFrame으로 저장
    metric_df = pd.DataFrame({
        'Class': ['Non-S', 'S'] * 4,
        'Metric': ['Precision'] * 2 + ['Recall'] * 2 + ['F1-Score'] * 2 + ['Specificity'] * 2,
        'Value': np.concatenate([precision, recall, f1, specificity])
    })
    metric_df['seed'] = seed
    
    return model, metric_df

def train_test_pipeline(x1, x2, y, records):
    all_metrics = []
    seed = HYPERPARAMS['seed']
    model_type = HYPERPARAMS['model_type']
    res_dir = os.path.join(RESULT_PATH, f"{EXPERIMENT_NAME}_seed{seed}_model{model_type}")
    os.makedirs(res_dir, exist_ok=True)
    print(f"Experiment: {EXPERIMENT_NAME}")

    train, val, test = split_data(x1, y, records, seed, x2)
    class_weights = compute_class_weights(train['y'])
    # S 클래스 가중치 수동 조정 (예: 5배)
    if 'S' in class_weights:
        class_weights['S'] *= 5
    y_train_oh, y_val_oh, y_test_oh, class_names = one_hot_encode(train['y'], val['y'], test['y'])
    model, metric_df = train_and_evaluate(
        model_type, train, val, test,
        y_train_oh, y_val_oh, y_test_oh, class_names, class_weights, seed
    )
    all_metrics.append(metric_df)
    # Save metrics
    metric_df.to_csv(os.path.join(res_dir, "metrics.csv"), index=False)
    # Save hyperparams
    export_hyperparams(HYPERPARAMS, os.path.join(res_dir, "hyperparams.json"))

def train_test_pipeline_binary(x1, x2, y, records):
    """Binary classification pipeline for S vs Non-S with configurable class weights"""
    all_metrics = []
    seed = HYPERPARAMS['seed']
    model_type = HYPERPARAMS['model_type']
    res_dir = os.path.join(RESULT_PATH, EXPERIMENT_NAME)
    os.makedirs(res_dir, exist_ok=True)
    print(f"Experiment: Binary S vs Non-S Classification")

    # Split data
    train, val, test = split_data(x1, y, records, seed, x2)
    
    # class weight 설정
    class_weight = None
    if HYPERPARAMS['use_class_weight']:
        class_weight = compute_class_weights(train['y'])
        # S 클래스 가중치 수동 조정
        if 1 in class_weight:  # binary classification에서는 S가 1로 인코딩됨
            class_weight[1] *= HYPERPARAMS['class_weight_multiply']
        print("Using class weights:", class_weight)
    else:
        print("Not using class weights")
    
    # Train and evaluate
    model, metric_df = train_and_evaluate_binary(
        model_type, train, val, test,
        train['y'], val['y'], test['y'],
        class_weight, seed
    )
    
    all_metrics.append(metric_df)
    
    # Save metrics
    metric_df.to_csv(os.path.join(res_dir, "metrics.csv"), index=False)
    # Save hyperparams
    export_hyperparams(HYPERPARAMS, os.path.join(res_dir, "hyperparams.json"))
    
    # train, val, test도 함께 반환
    return model, metric_df, train, val, test