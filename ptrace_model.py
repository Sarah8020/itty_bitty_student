import tensorflow as tf
import numpy as np
import energyusage

def create_preds():
    # ----open model
    student = tf.keras.models.load_model('student.model')
    
    # ----make test inputs
    test_set = []
    for i in range(100):
        test = [0] * 3000
        test_set.append(test)
    
    # ----make predictions
    predictions = student.predict(test_set)

if __name__ == '__main__':
    energyusage.evaluate(create_preds)
