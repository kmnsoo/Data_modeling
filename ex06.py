# 여러개의 레이어를 가진 모델

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#x(인풋) 값과 정답(y) 값 그리고 x2값  데이터 설정
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([11,12,13,14,15])

# 순차적 모델 설정 
model = Sequential()
#인풋레이어 노드10개와 입력층의 차원 1 활성화 함수는 relu로 주었다.
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))
 
#만들어진 모델 레이어 컴파일. 손실함수 mse, 가중치 최적화 adam, 메트릭스는 accuracy를 주었다.
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#데이터 학습. 학습 횟수는 100, 학습 데이터 사이즈는 1로 주어 학습 시킨다.
hist = model.fit(x, y, epochs=100, batch_size=1)
# 손실 함수 
mse= model.evaluate(x, y, batch_size=1)
print('mse : ', mse)
#예측 함수. 앞에 선언한 x2 데이터를 이용하여 예측
y_predict = model.predict(x2)
print("predict: \n", y_predict)


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
