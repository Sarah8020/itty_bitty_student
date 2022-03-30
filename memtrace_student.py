import tensorflow as tf
import numpy as np
from numpy import load
from os import listdir
import tracemalloc

if __name__ == '__main__':
    tracemalloc.start()
    # initialize testing data
    true_file = 'eeg_fpz_cz/' + 'EP_PSG_021921_EE141_01id0.npz'
    true_data = load(true_file)
    x_train = true_data['x']
    predict_file = 'teacher_predict/' + 'pred_EP_PSG_021921_EE141_01id0.npz'
    predict_data = load(predict_file)
    y_train = predict_data['y_pred']

    overall_matches = 0
    # get all training file names
    file_list = listdir('eeg_fpz_cz/')
    # go through files and get training data
    for item in file_list:
        if not 'npz' in item:
            continue
        if item == 'EP_PSG_021921_EE141_01id0.npz':
            continue
        # use teacher predictions as soft targets
        pred_file = 'teacher_predict/' + 'pred_' + item
        pred_data = load(pred_file)
        y_train = np.concatenate((y_train, pred_data['y_pred']))
        # get x training data
        train_file = 'eeg_fpz_cz/' + item
        train_data = load(train_file)
        x_train = np.concatenate((x_train, train_data['x']))

    # create student model
    tf.random.set_seed(0)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(.05))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))
    
    # train
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4)
    
    peak_mem = tracemalloc.get_traced_memory()[1]
    print('peak mem usage:', peak_mem, 'bytes')
    tracemalloc.stop()
