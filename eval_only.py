import student
import tensorflow as tf
import keras
import numpy as np
from numpy import load

if __name__ == '__main__':
    # ----open model
    model = tf.keras.models.load_model('student.model')
    student.test_model(model)
