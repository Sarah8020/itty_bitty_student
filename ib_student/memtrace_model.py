import tensorflow as tf
import numpy as np
from numpy import load
import tracemalloc

if __name__ == '__main__':
    tracemalloc.start()
    # ----open model
    student = tf.keras.models.load_model('student.model')
    
    # ----make test inputs
    test_set = []
    for i in range(1000):
        test = [0] * 3000
        test_set.append(test)
    
    # ----make predictions
    predictions = student.predict(test_set)
    # get prediction generation time
    peak_mem = tracemalloc.get_traced_memory()[1]
    print('peak mem usage:', peak_mem, 'bytes')
    tracemalloc.stop()
    
    # ----print some predictions
    print('predictions:')
    print(np.argmax(predictions[0]))
    print(np.argmax(predictions[1]))
