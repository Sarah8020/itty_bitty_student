import tensorflow as tf
import numpy as np
from numpy import load

if __name__ == '__main__':
    # open model
    student = tf.keras.models.load_model('student.model')
    
    # make 2 test inputs
    test_set = []
    test = []
    for i in range (3000):
        test.append(0.0)
    test2 = []
    for i in range (1500):
        test2.append(10.0)
        test2.append(-10.0)
    test_set.append(test)
    test_set.append(test2)
    
    # converting to numpy not neccessary
    #np_set = np.array(test_set)
    predictions = student.predict(test_set)
    
    print('predictions:')
    
    print(np.argmax(predictions[0]))
    print(np.argmax(predictions[1]))
