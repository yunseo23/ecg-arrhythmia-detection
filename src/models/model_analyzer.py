import numpy as np
import pandas as pd
from tensorflow.keras import Model
from .model import CNNx1OnlyModel, BinaryCNNAttentionModel


def calculate_layer_params(layer):
    """레이어의 파라미터 수 계산"""
    return layer.count_params()


def calculate_layer_ops(layer, input_shape):
    """레이어의 연산량 추정 (FLOPs)"""
    layer_type = layer.__class__.__name__
    
    if layer_type == 'Conv1D':
        # Conv1D: input_length * kernel_size * input_channels * output_channels
        kernel_size = layer.kernel_size[0]
        input_channels = layer.input_spec.axes[-1] if hasattr(layer.input_spec, 'axes') else input_shape[-1]
        output_channels = layer.filters
        output_length = input_shape[1]  # assuming same padding
        ops = output_length * kernel_size * input_channels * output_channels
        return ops
    
    elif layer_type == 'Dense':
        input_size = layer.input_spec.axes[-1] if hasattr(layer.input_spec, 'axes') else input_shape[-1]
        output_size = layer.units
        ops = input_size * output_size
        return ops
    
    elif layer_type == 'BatchNormalization':
        # BatchNorm: 4 operations per element (mean, var, normalize, scale)
        return np.prod(input_shape[1:]) * 4
    
    elif layer_type in ['GlobalAveragePooling1D', 'MaxPooling1D']:
        return np.prod(input_shape[1:])
    
    elif layer_type == 'MultiHeadSelfAttention':
        # Simplified attention computation
        seq_len = input_shape[1]
        d_model = input_shape[2]
        num_heads = getattr(layer, 'num_heads', 8)
        # Q*K^T + softmax + attention*V
        ops = seq_len * seq_len * d_model * num_heads * 3
        return ops
    
    else:
        return 0


def get_output_shape(layer, input_shape):
    """레이어의 출력 shape 계산"""
    layer_type = layer.__class__.__name__
    
    if layer_type == 'Conv1D':
        if layer.padding == 'same':
            return (input_shape[0], input_shape[1], layer.filters)
        else:
            # valid padding
            output_length = input_shape[1] - layer.kernel_size[0] + 1
            return (input_shape[0], output_length, layer.filters)
    
    elif layer_type == 'MaxPooling1D':
        pool_size = layer.pool_size[0]
        output_length = input_shape[1] // pool_size
        return (input_shape[0], output_length, input_shape[2])
    
    elif layer_type == 'GlobalAveragePooling1D':
        return (input_shape[0], input_shape[2])
    
    elif layer_type == 'Dense':
        return (input_shape[0], layer.units)
    
    elif layer_type in ['BatchNormalization', 'Activation', 'Dropout']:
        return input_shape
    
    elif layer_type == 'Add':
        return input_shape
    
    elif layer_type == 'MultiHeadSelfAttention':
        return input_shape
    
    else:
        return input_shape


def analyze_model_architecture(model, input_shape, model_name="Model"):
    """모델 아키텍처 분석하여 테이블 생성"""
    
    analysis_data = []
    current_shape = (None,) + input_shape
    
    def safe_format_shape(shape):
        """Shape을 안전하게 포맷팅"""
        if len(shape) <= 2:
            return f"1×{shape[-1]}"
        else:
            return f"{shape[1]}×{shape[2]}"
    
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        layer_name = layer.name
        
        # 특별한 케이스 처리
        if layer_type == 'InputLayer':
            layer_info = {
                'layer_name': 'input',
                'type': 'input',
                'kernel_size_stride': '-',
                'output_size': f"{input_shape[0]}×{input_shape[1]}",
                'depth': 0,
                'filters': '-',
                'params': 0,
                'ops': 0
            }
            current_shape = (None,) + input_shape
        
        elif layer_type == 'Conv1D':
            kernel_size = layer.kernel_size[0]
            stride = layer.strides[0]
            filters = layer.filters
            
            output_shape = get_output_shape(layer, current_shape)
            params = calculate_layer_params(layer)
            ops = calculate_layer_ops(layer, current_shape)
            
            layer_info = {
                'layer_name': layer_name,
                'type': 'conv1d',
                'kernel_size_stride': f"{kernel_size}×1/{stride}",
                'output_size': f"{output_shape[1]}×{output_shape[2]}",
                'depth': 1,
                'filters': filters,
                'params': params,
                'ops': ops
            }
            current_shape = output_shape
        
        elif layer_type == 'BatchNormalization':
            params = calculate_layer_params(layer)
            ops = calculate_layer_ops(layer, current_shape)
            
            layer_info = {
                'layer_name': layer_name,
                'type': 'batch norm',
                'kernel_size_stride': '-',
                'output_size': safe_format_shape(current_shape),
                'depth': 0,
                'filters': '-',
                'params': params,
                'ops': ops
            }
        
        elif layer_type == 'MaxPooling1D':
            pool_size = layer.pool_size[0]
            stride = layer.strides[0]
            
            output_shape = get_output_shape(layer, current_shape)
            
            layer_info = {
                'layer_name': layer_name,
                'type': 'max pool',
                'kernel_size_stride': f"{pool_size}×1/{stride}",
                'output_size': f"{output_shape[1]}×{output_shape[2]}",
                'depth': 0,
                'filters': '-',
                'params': 0,
                'ops': calculate_layer_ops(layer, current_shape)
            }
            current_shape = output_shape
        
        elif layer_type == 'GlobalAveragePooling1D':
            output_shape = get_output_shape(layer, current_shape)
            
            layer_info = {
                'layer_name': layer_name,
                'type': 'global avg pool',
                'kernel_size_stride': f"{current_shape[1]}×1/1",
                'output_size': f"1×{output_shape[1]}",
                'depth': 0,
                'filters': '-',
                'params': 0,
                'ops': calculate_layer_ops(layer, current_shape)
            }
            current_shape = output_shape
        
        elif layer_type == 'Dense':
            units = layer.units
            params = calculate_layer_params(layer)
            ops = calculate_layer_ops(layer, current_shape)
            
            output_shape = get_output_shape(layer, current_shape)
            
            layer_info = {
                'layer_name': layer_name,
                'type': 'dense',
                'kernel_size_stride': '-',
                'output_size': f"1×{units}",
                'depth': 1,
                'filters': '-',
                'params': params,
                'ops': ops
            }
            current_shape = output_shape
        
        elif layer_type == 'Dropout':
            rate = layer.rate
            
            layer_info = {
                'layer_name': layer_name,
                'type': f'dropout ({int(rate*100)}%)',
                'kernel_size_stride': '-',
                'output_size': safe_format_shape(current_shape),
                'depth': 0,
                'filters': '-',
                'params': 0,
                'ops': 0
            }
        
        elif layer_type == 'Activation':
            activation = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
            
            layer_info = {
                'layer_name': layer_name,
                'type': f'activation ({activation})',
                'kernel_size_stride': '-',
                'output_size': safe_format_shape(current_shape),
                'depth': 0,
                'filters': '-',
                'params': 0,
                'ops': 0
            }
        
        elif 'MultiHeadSelfAttention' in layer_type:
            params = calculate_layer_params(layer)
            ops = calculate_layer_ops(layer, current_shape)
            num_heads = getattr(layer, 'num_heads', 8)
            
            layer_info = {
                'layer_name': layer_name,
                'type': 'multi-head attn',
                'kernel_size_stride': '-',
                'output_size': safe_format_shape(current_shape),
                'depth': f'{num_heads} heads',
                'filters': '-',
                'params': params,
                'ops': ops
            }
        
        elif layer_type == 'Add':
            layer_info = {
                'layer_name': layer_name,
                'type': 'add (residual)',
                'kernel_size_stride': '-',
                'output_size': safe_format_shape(current_shape),
                'depth': 0,
                'filters': '-',
                'params': 0,
                'ops': 0
            }
        
        else:
            # 기타 레이어들
            layer_info = {
                'layer_name': layer_name,
                'type': layer_type.lower(),
                'kernel_size_stride': '-',
                'output_size': safe_format_shape(current_shape),
                'depth': 0,
                'filters': '-',
                'params': calculate_layer_params(layer),
                'ops': 0
            }
        
        analysis_data.append(layer_info)
    
    # DataFrame 생성
    df = pd.DataFrame(analysis_data)
    
    # 총합 계산
    total_params = df['params'].sum()
    total_ops = df['ops'].sum()
    
    return df, total_params, total_ops


def format_number(num):
    """숫자를 K, M 단위로 포맷팅"""
    if num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


def print_model_table(df, total_params, total_ops, model_name):
    """모델 테이블을 출력"""
    print(f"\n{'='*80}")
    print(f"Table: {model_name} Architecture")
    print(f"{'='*80}")
    
    # 파라미터와 ops를 포맷팅
    df_display = df.copy()
    df_display['params'] = df_display['params'].apply(lambda x: format_number(x) if x > 0 else '0')
    df_display['ops'] = df_display['ops'].apply(lambda x: format_number(x) if x > 0 else '0')
    
    # 테이블 출력
    print(df_display.to_string(index=False))
    
    print(f"\nTotal Parameters: {format_number(total_params)}")
    print(f"Total Operations: {format_number(total_ops)}")
    print(f"{'='*80}")


def analyze_cnnx1only_model(x1_shape=(300, 1), n_classes=4):
    """CNNx1OnlyModel 분석"""
    model = CNNx1OnlyModel(x1_shape=x1_shape, n_classes=n_classes)
    df, total_params, total_ops = analyze_model_architecture(
        model.model, x1_shape, "CNNx1OnlyModel"
    )
    print_model_table(df, total_params, total_ops, "CNNx1OnlyModel")
    return df, total_params, total_ops


def analyze_binary_cnn_attention_model(input_shape=(300, 1)):
    """BinaryCNNAttentionModel 분석"""
    model = BinaryCNNAttentionModel(input_shape=input_shape)
    df, total_params, total_ops = analyze_model_architecture(
        model.model, input_shape, "BinaryCNNAttentionModel"
    )
    print_model_table(df, total_params, total_ops, "BinaryCNNAttentionModel")
    return df, total_params, total_ops


def analyze_all_models():
    """모든 모델 분석"""
    print("Analyzing ECG Classification Models...")
    
    # CNNx1OnlyModel 분석
    df1, params1, ops1 = analyze_cnnx1only_model()
    
    # BinaryCNNAttentionModel 분석  
    df2, params2, ops2 = analyze_binary_cnn_attention_model()
    
    # 비교 요약
    print(f"\n{'='*80}")
    print("Model Comparison Summary")
    print(f"{'='*80}")
    comparison_data = {
        'Model': ['CNNx1OnlyModel', 'BinaryCNNAttentionModel'],
        'Input Shape': ['(300, 1)', '(300, 1)'],
        'Output': ['4 classes', 'Binary'],
        'Parameters': [format_number(params1), format_number(params2)],
        'Operations': [format_number(ops1), format_number(ops2)],
        'Key Features': ['Simple CNN + Global Pooling', 'Residual blocks + Multi-head Attention']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print(f"{'='*80}")


if __name__ == "__main__":
    analyze_all_models()
