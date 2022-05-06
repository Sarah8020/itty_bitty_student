# ib_private
The Itty Bitty model public repo. These scripts will not run properly without Earable data. Earable data is not public.  
Running "python3 student.py" will train, evaluate, and save the student model.  
Running "python3 eval_only.py" will evaluate the student and provide a confusion matrix.  
Careful - only use "python3 eval_only.py" so as not to overwrite the model. Different hardware/ software platforms can output different models.  
If you have access to the private data storage, place "student_test_set.npz" and "student_train_set.npz" into the same directory as the student.py and eval_only scripts. These can be found in the Student directory inside the Team2 directory. The train data is ground truth x data with modified TSN's y outputs. The test data is ground truth x and y data. Tip: It's much faster to load lots of data from one .npz file than several smaller files (which requires array/list concatonation).  
Student.py and eval_only.py can then be used.  
