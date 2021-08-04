# 데이터 셋 분류 
# 준비 데이터를 제외한 내부 코드가 동일하기에 주석 달지 않음.
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
 
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# 데이터 셋을 train과 test로 분류
x_predict = np.array([21,22,23,24,25,26,27,28,29,30])
 
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))
 
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=2)
 
loss, acc = model.evaluate(x_test, y_test, batch_size=2)
print('acc : ', acc)
print('loss : ', loss)
 
y_predict = model.predict(x_predict)
print(y_predict)

#손실함수 R2이용. 이전에는 mse를 이용했다.
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)






#import matplotlib.pyplot as plt
#import numpy as np#
##matplotlib를 활용하면 하나의 그래프로 배열 값들을 쉽게 표시 가능.
#fig, loss_ax = plt.subplots()
#acc_ax = loss_ax.twinx()
#loss_ax.plot(hist.history['loss'], 'y', label='train loss')
##loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
#acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
##acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
#loss_ax.set_xlabel('epoch')
#loss_ax.set_ylabel('loss')
#acc_ax.set_ylabel('accuray')
#loss_ax.legend(loc='upper left')
#acc_ax.legend(loc='lower left')
#plt.show()