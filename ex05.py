from keras.models import Sequential
from keras.layers import Dense

import numpy as np
# 랜덤 시드 고정 값 3설정
np.random.seed(3)


x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
 
model = Sequential()
# input_dim=1은 입력 차원이 1이라는 뜻. 입력 노드가 1개라고 생각하면 됨.
# 만약 x배열의 데이터가 2개라면 2, 3개라면 3으로 지정해주면 된다.
# Dense 바로 뒤 숫자 5, 3, 1은 노드의 개수
# relu는 활성화 함수 종류 중 하나로 은닉 층으로 학습을 의미한다.
model.add(Dense(5, input_dim=1, activation='relu'))
# add를 통해 레이어를 추가해줄 수 있다.
model.add(Dense(3))
# dense레이어는 입력과 출력을 모두 연결해주며 입력과 출력을 각각 연결해주는 가중치를 포함한다
# 입력이 3개 출력이 4개라면 가중치는 총 3X4인 12개가 존재하게 됩니다. Dense레이어는 머신러닝의 기본층

#1개의 레이어.
model.add(Dense(1))
#모델 컴파일. 손실함수 mse, 가중치 최적화 옵티마이저는 adam으로
model.compile(loss='mse', optimizer='adam')
#모델 학습. 학습 횟수는 100, 학습 데이터 사이즈는 1로 설정 후 변수 hist에 담았다.
hist = model.fit(x, y, epochs=100, batch_size=1)
#손실 함수 정의. 
mse = model.evaluate(x, y, batch_size=1)
 
print('mse : ', mse)

import matplotlib.pyplot as plt
import numpy as np#
#matplotlib를 활용하면 하나의 그래프로 배열 값들을 쉽게 표시 가능.
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
#loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
#acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()