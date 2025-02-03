import tensorflow as tf
from tensorflow import keras
import numpy as np

def weighted_focal_loss(y_true, y_pred):
    # Class weights from your calculation
    class_weights = tf.constant([[5.23, 56.97, 9.21, 162.23, 9.89, 52.80]])
    
    # Focal loss parameters
    gamma = 2.0
    alpha = 0.25
    
    epsilon = keras.backend.epsilon()
    y_pred = keras.backend.clip(y_pred, epsilon, 1.0-epsilon)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    
    focal_loss = -keras.backend.sum(
        alpha * keras.backend.pow(1 - pt, gamma) * keras.backend.log(pt),
        axis=-1
    )
    
    return keras.backend.mean(focal_loss * class_weights, axis=-1)

# Test the loss function
if __name__ == "__main__":
    # Create sample data
    y_true = tf.constant([[1, 0, 1, 0, 1, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0.7, 0.1, 0.8, 0.2, 0.9, 0.1]], dtype=tf.float32)
    
    # Calculate loss
    loss = weighted_focal_loss(y_true, y_pred)
    print("Test loss value:", loss.numpy())

