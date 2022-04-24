import student
import tensorflow as tf
import keras
import numpy as np
from numpy import load

if __name__ == '__main__':
    # ----open model
    model = tf.keras.models.load_model('student.model')
    y_pred, y_true, f1, acc = student.test_model(model)
    
    pred_len = len(y_pred)
    conf = []
    for i in range(4):
        row = [0] * 4
        conf.append(row)
    
    for i in range(pred_len):
        pred = y_pred[i]
        truth = y_true[i]
        conf[truth][pred] += 1
        
    conf_prob = [x[:] for x in conf]
    for i, row in enumerate(conf):
        total = sum(row)
        for j, num in enumerate(row):
            conf_prob[i][j] = num/total
    
    print('confusion matrix with raw numbers:')
    for row in conf:
        print(row)
    print('confusion matrix with probabilities:')
    for row in conf_prob:
        print(row)
        
