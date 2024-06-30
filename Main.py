from zipfile import ZipFile
import os 
import urllib
import urllib.request
import numpy as np
import cv2


#@ Import Dataset from Website:
URL='https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE='fashion_mnist_images.zip'
FOLDER= 'fashion_mnist_images'


if not os.path.isfile(FILE):
    print(f'Downloading {URL} and Saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)

# print('Unzipping images....')

with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)

# print('Done!')

#@ Loading MNIST Fashion Dataset:

def load_dataset(dataset, path):
    labels=os.listdir(os.path.join(path, dataset))

    #creating list for samples and labels:
    X=[]
    y=[]

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image=cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)

    
    # Converting the data to proper numpy arrays and return:
    return np.array(X), np.array(y).astype('uint8')

#@ MNIST dataset(train + test)
def create_data_mnist(path):
    X, y=load_dataset('train', path)
    X_test, y_test=load_dataset('test', path)

    return X, y, X_test, y_test

#@ Creating dataset:
X, y, X_test, y_test=create_data_mnist('fashion_mnist_images')

#@ Shuffling the traininmg dataset:
keys=np.array(range(X.shape[0]))
np.random.shuffle(keys)
X=X[keys]
y=y[keys]

#@ Scaling & Reshaping the vectors:(Converting into 1D array)
X=(X.reshape(X.shape[0], -1).astype(np.float32)- 127.5)/127.5
X_test=(X_test.reshape(X_test.shape[0], -1).astype(np.float32)-127.5)/127.5


