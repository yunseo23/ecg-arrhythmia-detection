import tensorflow as tf
import os
from tensorflow.keras import backend as K
from .model import CNNx1OnlyModel, BinaryCNNAttentionModel


def export_model_for_netron(model, filepath, input_shape=None, sample_input=None):
    """
    Export a model in formats that can be visualized in Netron
    
    Args:
        model: Keras model instance
        filepath: Path to save the model (without extension)
        input_shape: Input shape for the model (optional, used for SavedModel format)
        sample_input: Sample input data for the model (optional)
    """
    base_path = os.path.dirname(filepath)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Clean up layer names for better visualization
    
    # Save as Keras model (.keras format - recommended)
    keras_path = f"{filepath}.keras"
    model.save(keras_path)
    print(f"Model saved as Keras format: {keras_path}")
    
    # Save as SavedModel format (also supported by Netron)
    # In Keras 3, we save to a directory without extension for SavedModel format
    savedmodel_path = f"{filepath}_savedmodel"
    try:
        # Use tf.saved_model.save for SavedModel format
        tf.saved_model.save(model, savedmodel_path)
        print(f"Model saved as SavedModel format: {savedmodel_path}")
    except Exception as e:
        print(f"Warning: Could not save as SavedModel format: {e}")
    
    # Save as HDF5 format (.h5)
    h5_path = f"{filepath}.h5"
    try:
        model.save(h5_path)
        print(f"Model saved as HDF5 format: {h5_path}")
    except Exception as e:
        print(f"Warning: Could not save as HDF5 format: {e}")





def export_cnnx1only_model(x1_shape=(300, 1), n_classes=4, output_dir="exported_models", clear_session=True):
    """
    Export CNNx1OnlyModel for Netron visualization
    
    Args:
        x1_shape: Input shape for the time series data
        n_classes: Number of output classes
        output_dir: Directory to save the exported models
        clear_session: Whether to clear Keras session before creating model
    """
    if clear_session:
        K.clear_session()
        print("Keras session cleared")
    
    print("Creating CNNx1OnlyModel...")
    model = CNNx1OnlyModel(x1_shape=x1_shape, n_classes=n_classes)
    
    # Create sample input for testing
    import numpy as np
    sample_input = np.random.random((1,) + x1_shape).astype(np.float32)
    
    # Test the model with sample input
    _ = model.model.predict(sample_input)
    print("Model created and tested successfully")
    
    # Export the model
    filepath = os.path.join(output_dir, "cnnx1only_model")
    export_model_for_netron(model.model, filepath, x1_shape, sample_input)
    
    return model


def export_binary_cnn_attention_model(input_shape=(300, 1), output_dir="exported_models", clear_session=False):
    """
    Export BinaryCNNAttentionModel for Netron visualization
    
    Args:
        input_shape: Input shape for the model
        output_dir: Directory to save the exported models
        clear_session: Whether to clear Keras session before creating model
    """
    if clear_session:
        K.clear_session()
        print("Keras session cleared")
        
    print("Creating BinaryCNNAttentionModel...")
    model = BinaryCNNAttentionModel(input_shape=input_shape)
    
    # Create sample input for testing
    import numpy as np
    sample_input = np.random.random((1,) + input_shape).astype(np.float32)
    
    # Test the model with sample input
    _ = model.model.predict(sample_input)
    print("Model created and tested successfully")
    
    # Export the model
    filepath = os.path.join(output_dir, "binary_cnn_attention_model")
    export_model_for_netron(model.model, filepath, input_shape, sample_input)
    
    return model


def export_all_models(x1_shape=(300, 1), n_classes=4, output_dir="exported_models"):
    """
    Export all specified models for Netron visualization
    
    Args:
        x1_shape: Input shape for CNNx1OnlyModel
        n_classes: Number of classes for CNNx1OnlyModel
        output_dir: Directory to save all exported models
    """
    print("Exporting all models for Netron visualization...")
    print("=" * 50)
    
    # Export CNNx1OnlyModel
    cnn_model = export_cnnx1only_model(x1_shape, n_classes, output_dir, clear_session=True)
    print()
    
    # Export BinaryCNNAttentionModel
    attention_model = export_binary_cnn_attention_model(x1_shape, output_dir, clear_session=True)
    print()
    
    print("All models exported successfully!")
    print(f"Models saved in: {output_dir}")
    print("\nTo visualize in Netron:")
    print("1. Install Netron: pip install netron")
    print("2. Run: netron <model_file_path>")
    print("3. Supported formats: .keras, .h5, SavedModel folder")
    
    return cnn_model, attention_model


if __name__ == "__main__":
    # Example usage
    export_all_models()
