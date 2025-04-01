from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def find_closest_value(data:dict, target):
    # 각 key와 target의 차이가 가장 작은 key 선택
    closest_key = min(data.keys(), key=lambda k: abs(k - target))
    return data[closest_key]
    
def extract_labels(rpeaks, symbols):
    labels = []
    for rpeak in rpeaks:
        label = find_closest_value(symbols, rpeak)
        labels.append(label)
    return labels
    
def group_labels(label):
    if label in ['N','L','R','e','j']:
        return 'N' # N (Normal)
    elif label in ['A','a','J','S']:  
        return 'S' # S (Supraventricular ectopic)
    elif label in ['V','E']:  
        return 'V' # V (Ventricular ectopic)
    elif label in ['/','f','F','Q']:
        return 'Q' # Q (Unknown/Paced)
    else:
        return 'O' # other
    

def one_hot_encoder(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # label 값 정수로 변환
    y_onehot = to_categorical(y_encoded)  # One-hot encoding 수행

    # class_name classification report에 matching하기위해 저장
    class_names = le.classes_

    return y_onehot, class_names