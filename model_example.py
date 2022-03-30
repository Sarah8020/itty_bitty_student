import tensorflow as tf
import numpy as np
from numpy import load
import time

if __name__ == '__main__':
    # ----open model
    student = tf.keras.models.load_model('student.model')
    
    # ----make test inputs
    test_set = []
    for i in range(100):
        test = [0] * 3000
        test_set.append(test)
    
    # ----make predictions
    start_time = time.time()
    predictions = student.predict(test_set)
    # get prediction generation time
    print('time to generate 100 predictions: ', time.time() - start_time)
    
    # ----print some predictions
    print('predictions:')
    print(np.argmax(predictions[0]))
    print(np.argmax(predictions[1]))
