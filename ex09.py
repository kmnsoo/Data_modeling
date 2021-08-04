
#slice Notion 사용해서 데이터 자르기(나누기)

# 1. 데이터 나누어 준비. 
import numpy as np
x = np.array(range(1,101)) # 1~100
y = np.array(range(1,101))
print(x)
x_train = x[:60] # 1~60
x_val = x[60:80] # 61~80
x_test = x[80:]  # 81~100
y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]
 
# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_shape=(1, ), activation='relu'))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(1))
 
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


