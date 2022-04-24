import tensorflow as tf
import keras
import numpy as np
from numpy import load
from os import listdir
import time
from sklearn.utils import class_weight

def main():
    # initialize testing data
    true_file = 'eeg_fpz_cz/' + 'EP_PSG_021921_EE141_01id0.npz'
    true_data = load(true_file)
    x_train = true_data['x']
    y_true_labels = true_data['y']
    predict_file = 'teacher_predict/' + 'pred_EP_PSG_021921_EE141_01id0.npz'
    predict_data = load(predict_file)
    y_train = predict_data['y_pred']
    
    # get all training file names
    file_list = listdir('eeg_fpz_cz/')
    # go through files and get training data
    for item in file_list:
        if not 'npz' in item:
            continue
        if item == 'EP_PSG_021921_EE141_01id0.npz':
            continue
        print('getting ', item)
        pred_file = 'teacher_predict/' + 'pred_' + item
        pred_data = load(pred_file)
        #accuracy = get_pred_accuracy(pred_data['y_pred'], pred_data['y_true'])
        #print('individual file accuracy:', item, str(accuracy))
        # use teacher predictions as soft targets
        y_train = np.concatenate((y_train, pred_data['y_pred']))
        # get x training data
        train_file = 'eeg_fpz_cz/' + item
        train_data = load(train_file)
        x_train = np.concatenate((x_train, train_data['x']))
        y_true_labels = np.concatenate((y_true_labels, pred_data['y_true']))
    
    accuracy = get_pred_accuracy(y_train, y_true_labels)
    print('tsn preds overall accuracy:', str(accuracy))

    print('x training length:', str(len(x_train)))
    
    label_true_counts = [0,0,0,0]
    label_pred_counts = [0,0,0,0]
    correct_preds = [0,0,0,0]
    for i in range(len(y_train)):
        label_pred_counts[y_train[i]] += 1
        label_true_counts[y_true_labels[i]] += 1
        if y_train[i] == y_true_labels[i]:
            correct_preds[y_train[i]] += 1
    for num in [0,1,2,3]:
        print(num,'occurance info:', len(y_train), label_pred_counts[num], label_pred_counts[num]/len(y_train))
    
    # create student model
    tf.random.set_seed(0)
    np.random.seed(0)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(.05))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))
    
    start_train_time = time.time()
    # train
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy()])
    # automated weight calculation with sklearn produced worse results
    training_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    training_weight = dict(enumerate(training_weight))
    print('automated weights:', training_weight)
    model.fit(x_train, y_train, epochs=10, steps_per_epoch=200, shuffle=True)
    print('training time:', time.time() - start_train_time, 'seconds')
    
    # test
    test_model(model)
    
    # model info
    #model.summary()
    
    save_model(model)

def save_model(model):
    # save model
    model.save('student.model')
    conv = tf.lite.TFLiteConverter.from_saved_model('student.model')
    lite_student_model = conv.convert()
    with open('student_model.tflite', 'wb') as lite_file:
        lite_file.write(lite_student_model)

# get precision for student
def get_prec(this_class, y_preds, y_trues):
    true_count = 0
    for num in y_preds:
        if num == this_class:
            true_count += 1
    correct_preds = 0
    for i in range(len(y_preds)):
        if y_preds[i] == this_class and y_trues[i] == this_class:
            correct_preds += 1
    if true_count == 0:
        return 0, 0
    return (correct_preds / true_count), correct_preds

# get recall for student
def get_recall(this_class, y_preds, y_trues, correct_preds):
    false_negs = 0
    for i in range(len(y_preds)):
        if y_preds[i] != this_class and y_trues[i] == this_class:
            false_negs += 1
    if (correct_preds + false_negs) < 0.000001:
        return 0
    return correct_preds / (correct_preds + false_negs)

# get accuracy for a single prediction file
def get_pred_accuracy(preds, reals):
    matches = 0
    for i in range(len(preds)):
        if preds[i] == reals[i]:
            matches += 1
    accuracy = matches / len(preds)
    return accuracy

def test_model(model):
    # get some unseen testing data
    eval_test_file = 'eeg_no_pred/' + 'EP_PSG_031521_EE213id31.npz'
    eval_test_data = load(eval_test_file)
    x_test = eval_test_data['x']
    y_test = eval_test_data['y']
    # evaluate using testing data
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print('loss on unseen data set:', str(val_loss))
    print('accuracy on unseen data set:', str(val_acc))
    
    # get y preds for x_test
    predictions = model.predict(x_test)
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
    avg_f1 = sum(f1s)/4
    print('avg f1:', avg_f1)
    print('len of preds:', len(y_preds))
    
    return y_preds, y_test
    
def reduce_training_size(x_train, y_train, max_size):
    y_counts = [0,0,0,0]
    short_x_train = []
    short_y_train = []
    for i, label in enumerate(y_train):
        if y_counts[label] > max_size:
            continue
        y_counts[label] += 1
        short_x_train.append(x_train[i])
        short_y_train.append(label)
    x_train = np.array(short_x_train)
    y_train = np.array(short_y_train)
    return x_train, y_train

if __name__ == '__main__':
    main()
