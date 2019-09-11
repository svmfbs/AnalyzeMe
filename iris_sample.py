from subprocess import check_output

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print ('tf       :', tf.version)
    print ('tf.teras :', tf.keras.__version__)
    print ('keras    :', keras.__version__)
    # print (check_output('ls', '.').decode('utf-8')) # error ???
    dataset = pd.read_csv('~/sample/Iris.csv')
    print (dataset.head())

if __name__ == "__main__":
    main()

