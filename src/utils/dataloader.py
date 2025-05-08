import numpy as np
import tensorflow as tf
from src.preprocess.augmentation import get_augmented_data
from config import HYPERPARAMS

def create_dataset(signals, labels, batch_size=32, shuffle=True, augmentation_config=None):
    """
    Create a TensorFlow Dataset for ECG signals with optional augmentation.
    
    Args:
        signals (np.ndarray): Array of ECG signals
        labels (np.ndarray): Array of corresponding labels
        batch_size (int): Batch size for the dataset
        shuffle (bool): Whether to shuffle the data
        augmentation_config (dict): Configuration for data augmentation
        
    Returns:
        tf.data.Dataset: TensorFlow Dataset instance
    """
    def augment_signal(signal, label):
        if augmentation_config is not None and label in augmentation_config:
            augmented_signals, augmented_labels = get_augmented_data(
                signal, 
                label, 
                augmentation_config[label]
            )
            # Randomly select one augmented version
            aug_idx = tf.random.uniform(shape=[], minval=0, maxval=len(augmented_signals), dtype=tf.int32)
            signal = augmented_signals[aug_idx]
            label = augmented_labels[aug_idx]
        return signal, label

    # Create dataset from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((signals, labels))
    
    # Apply augmentation if configured
    if augmentation_config is not None:
        dataset = dataset.map(
            lambda x, y: tf.py_function(
                func=augment_signal,
                inp=[x, y],
                Tout=[tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset 