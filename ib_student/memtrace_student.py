import tensorflow as tf
import numpy as np
from numpy import load
import tracemalloc

if __name__ == '__main__':
    tracemalloc.start()
    # initialize testing data
    training_data_file = load('student_train_set.npz')
    x_train = training_data_file['x']
    y_train = training_data_file['y']

    # create student model
    tf.random.set_seed(0)
    model = tf.keras.models.load_model('student.model')
    
    # train
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, shuffle=True, steps_per_epoch=200)
    
    peak_mem = tracemalloc.get_traced_memory()[1]
    print('peak mem usage:', peak_mem, 'bytes')
    tracemalloc.stop()
