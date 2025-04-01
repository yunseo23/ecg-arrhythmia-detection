from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd

def calc_metrics(y_test, y_pred, labels):
    def calc_specificity(y_test, y_pred, labels):
        conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(labels)))
        specificity = []

        for i in range(len(labels)):
            tn = np.sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1))  # TN: 행/열 삭제로 계산
            fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]  # FP: 해당 열의 합에서 TP를 제외
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        return specificity
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=range(len(labels)))
    specificity = calc_specificity(y_test, y_pred, labels)

    return precision, recall, f1, specificity

def get_metric_df(precision, recall, f1, specificity, class_names):
    # 결과를 DataFrame으로 정리
    data = {
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity,
    }
    metrics_df = pd.DataFrame(data)
    return metrics_df