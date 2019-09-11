""" This is regression sample program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
sys.dont_write_bytecode = True # dont make .pyc files

# Boston Housingのデータセットの読み込み
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep=r'\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
X = df.iloc[:, 0:13]
Y = df['MEDV'].values
print(X.head())

# データの整形
sc = preprocessing.StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

# 学習データとテストデータに分割する
X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.2, random_state=0)

# 線形回帰モデルの生成および学習
clf = linear_model.SGDRegressor(max_iter=1000)
clf.fit(X_train, Y_train)

print('回帰式の係数')
print('y切片:', clf.intercept_)
print('各項目の係数', clf.coef_)

# テストデータに対する2乗誤差の平均
Y_pred = clf.predict(X_test)
RMS = np.mean((Y_pred - Y_test)**2)
print('テストデータに対する2乗誤差の平均', RMS)

# 与えられたデータに対する予測
compare = pd.DataFrame(np.array([Y_test, Y_pred]).T)
compare.columns = ['正解', '予測値']
print(compare[:20])

# Kerasで回帰分析
# 前処理
x_train = X_train
x_test = X_test
y_train = Y_train
y_test = Y_test
model = Sequential()
model.add(Dense(1000, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

model.compile(Adam(lr=1e-3), loss='mean_squared_error')

# トレーニングデータで学習し、テストデータで評価(平均2乗誤差を用いる)
history = model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_test, y_test))
print(model.evaluate(x_test, y_test))

# 与えられたデータに対する予測
pred = model.predict(x_test).flatten()
compare = pd.DataFrame(np.array([y_test, pred]).T)
compare.columns = ['正解', '予測値']
print(compare[:20])

