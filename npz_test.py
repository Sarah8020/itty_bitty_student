import numpy as np
from numpy import load
from os import listdir

if __name__ == '__main__':
    print('true file format')
    
    true_file = 'eeg_fpz_cz/' + 'EP_PSG_021921_EE141_01id0.npz'
    true_data = load(true_file)
    print(true_data.files)
    
    print('prediction file format')
    
    pred_file = 'teacher_predict/' + 'pred_EP_PSG_021921_EE141_01id0.npz'
    pred_data = load(pred_file)
    print(pred_data.files)
