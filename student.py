import tensorflow as tf
import numpy as np
from numpy import load
from os import listdir

def get_pred_accuracy(preds, reals):
    matches = 0
    for i in range(len(preds)):
        if preds[i] == reals[i]:
            matches += 1
    accuracy = matches / len(preds)
    return accuracy
    
def get_tsn_accuracy(preds, reals, matches, y_length):
    for i in range(len(preds)):
        if preds[i] == reals[i]:
            matches += 1
    accuracy = matches / y_length
    print('tsn predictions overall accuracy: ', str(accuracy))
    return matches

if __name__ == '__main__':
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
        print('getting ', item)
        pred_file = 'teacher_predict/' + 'pred_' + item
        pred_data = load(pred_file)
        accuracy = get_pred_accuracy(pred_data['y_pred'], pred_data['y_true'])
        print('individual file accuracy:', item, str(accuracy))
        y_train = np.concatenate((y_train, pred_data['y_pred']))
        # get x training data
        train_file = 'eeg_fpz_cz/' + item
        train_data = load(train_file)
        x_train = np.concatenate((x_train, train_data['x']))
        # check overall tsn accuracy
        overall_matches = get_tsn_accuracy(pred_data['y_pred'], pred_data['y_true'], overall_matches, len(y_train))

    print('x training length:', str(len(x_train)))

    # create student model
    tf.random.set_seed(0)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    # 0.6559633016586304 (1.5mb tflite file)
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(.05))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))
    # 0.6605504751205444 (15.1mb tflite file)
    '''
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))
    '''
    
    # train
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4)
    
    # get unseen testing data
    eval_test_file = 'eeg_no_pred/' + 'EP_PSG_031021_EE193id20.npz'
    eval_test_data = load(eval_test_file)
    x_test = eval_test_data['x']
    y_test = eval_test_data['y']
    # evaluate
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print('loss on unseen data set:', str(val_loss))
    print('accuracy on unseen data set:', str(val_acc))
    
    # model info
    model.summary()

    # save model
    model.save('student.model')
    conv = tf.lite.TFLiteConverter.from_saved_model('student.model')
    lite_student_model = conv.convert()
    with open('student_model.tflite', 'wb') as lite_file:
        lite_file.write(lite_student_model)
