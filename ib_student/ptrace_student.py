import tensorflow as tf
import numpy as np
from numpy import load
import energyusage

def train():
    # initialize testing data
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
    model.fit(x_train, y_train, epochs=10, steps_per_epoch=200, shuffle=True)

if __name__ == '__main__':
    energyusage.evaluate(train)
