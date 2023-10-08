1.Requirements£º
Python 3.7 (Anaconda)
Tensorflow 1.14.0 or above
CUDA 10.0 or above
2.Datasets
Our code assumes you have prepared image dataset as following.
dataset
©À©¤©¤ test
©¦   ©À©¤©¤ bp_norm # a folder containing all the testing low-resolution images
©¦   ©¸©¤©¤ img_norm # a folder containing all the testing high-resolution images
©À©¤©¤ train
©¦  ©À©¤©¤ train_bp # a folder containing all the training low-resolution images
©¦  ©¸©¤©¤ train_label # a folder containing all the training high-resolution images
©¦   
©¸©¤©¤ val
    ©À©¤©¤ val_bp # a folder containing all the validated low-resolution images
    ©¸©¤©¤ val_label # a folder containing all the validated high-resolution images
3.Train and Test
python train.py
python test.py



