# 1. 데이터 2차원 데이터 준비
import numpy as np
x = np.array([range(1,101), range(101,201)]) # 데이터가 두개!
y = np.array([range(1,101), range(101,201)]) 
print(x) 
 
print(x.shape) # (2,100) 2행 100열
 
x = np.transpose(x)
y = np.transpose(y)
 
print(x.shape) # (100,2)
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5)
 
# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_shape=(2, ), activation='relu'))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(2)) # input col 2개 이므로 output도 2개
 
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))
 
# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', mse)
 
y_predict = model.predict(x_test)
print(y_predict)
 
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict)) 
 
# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)


