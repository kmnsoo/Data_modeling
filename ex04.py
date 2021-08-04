from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import numpy as np
from numpy.lib.npyio import save

#seed값을 같게 새팅하면 같은 난수 발생, 다르게 세팅하면 다른 난수 발생.
np.random.seed(3)

# 1. 데이터셋 준비하기

# 훈련셋과 시험셋 로딩
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 훈련셋과 검증셋 분리
X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]

#astype()메소드를 이용하여 문자열 컬럼의 숫자형 변환 가능
X_train = X_train.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(10000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

# 훈련셋, 검증셋 고르기
train_rand_idxs = np.random.choice(50000, 700)
val_rand_idxs = np.random.choice(10000, 300)

X_train = X_train[train_rand_idxs]
Y_train = Y_train[train_rand_idxs]
X_val = X_val[val_rand_idxs]
Y_val = Y_val[val_rand_idxs]
# 라벨링 전환 (원핫인코딩 처리)
Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=2, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
import pandas as pd
from matplotlib import pyplot as plt


# 3. 모델 엮기 (모델 학습 과정 설정하기)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
from keras.callbacks import EarlyStopping
# 조기종료 콜백함수 정의, patience => 로스가 증가되어도 에포크 20까지는 더 실행해본다.(기다린다)
early_stopping = EarlyStopping(patience = 20) 
# x : 입력 데이터 
# y : 라벨 값
# epochs : 학습 반복 횟수
# batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 지정
hist = model.fit(X_train, Y_train, epochs=3000, batch_size=10, validation_data=(X_val, Y_val), callbacks=[early_stopping])
model.save("model.h5")
from tensorflow.keras.models import load_model
model = load_model('model.h5')
