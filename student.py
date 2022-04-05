import tensorflow as tf
import keras
import numpy as np
from numpy import load
from os import listdir
import time

# get accuracy for a single prediction file
def get_pred_accuracy(preds, reals):
    matches = 0
    for i in range(len(preds)):
        if preds[i] == reals[i]:
            matches += 1
    accuracy = matches / len(preds)
    return accuracy
    
# get overall accuracy for prediction files
def get_tsn_accuracy(preds, reals, matches, y_length):
    for i in range(len(preds)):
        if preds[i] == reals[i]:
            matches += 1
    accuracy = matches / y_length
    print('tsn predictions overall accuracy: ', str(accuracy))
    return matches

# get precision for student
def get_prec(this_class, y_preds, y_trues):
    true_count = 0
    for num in y_trues:
        if num == this_class:
            true_count += 1
    correct_preds = 0
    for i in range(len(y_preds)):
        if y_preds[i] == this_class and y_trues[i] == this_class:
            correct_preds += 1
    return (correct_preds / true_count), correct_preds

# get recall for student
def get_recall(this_class, y_preds, y_trues, correct_preds):
    false_negs = 0
    for i in range(len(y_preds)):
        if y_preds[i] == this_class and y_trues[i] != this_class:
            false_negs += 1
    if (correct_preds + false_negs) < 0.000001:
        return 0
    return correct_preds / (correct_preds + false_negs)
    

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
    # 0.5563636422157288 accuracy
    # avg f1: 0.40243012120371624
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(.05))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))
    
    start_train_time = time.time()
    # train
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy()])
    model.fit(x_train, y_train, epochs=10, shuffle=True, steps_per_epoch=200)
    print('training time:', time.time() - start_train_time, 'seconds')
    
    # get unseen testing data
    eval_test_file = 'eeg_no_pred/' + 'EP_PSG_052721_EE502_01id169.npz'
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
        
    # get precision/ recall (per class)
    # get y preds for x_test
    student = tf.keras.models.load_model('student.model')
    predictions = student.predict(x_test)
    y_preds = []
    for i in range(len(predictions)):
        y_preds.append(np.argmax(predictions[i]))
    # get precision/ recall/ f1 score for each class
    all_ys = [0,1,2,3]
    f1s = []
    for num in all_ys:
        prec, correct_preds = get_prec(num, y_preds, y_test)
        recall = get_recall(num, y_preds, y_test, correct_preds)
        f1 = 0
        if prec+recall > 0.000001:
            f1 = 2*((prec*recall)/(prec+recall))
            f1s.append(f1)
        print('class:', num, 'precision:', prec, 'recall:', recall, 'f1:', f1)
    print('avg f1:', sum(f1s)/4)
    
    count_other = 0
    count_1 = 0
    test_count_other = 0
    test_count_1 = 0
    for i in range(len(y_preds)):
        if y_preds[i] == 1:
            count_1 += 1
        else:
            count_other += 1
        if y_test[i] == 1:
            test_count_1 += 1
        else:
            test_count_other += 1
    print('check if over predicting label 1 ----')
    print('# of 1 labels in preds:', count_1)
    print('# of other labels in preds', count_other)
    print('# of 1 labels in real data:', test_count_1)
    print('# of other labels in real data', test_count_other)
