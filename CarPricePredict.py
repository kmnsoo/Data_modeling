import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression

##########데이터 로드

train_df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/regression/carprice_E1SUl6b.xlsx?raw=true', sheet_name='train')
test_df = pd.read_excel('https://github.com/cranberryai/todak_todak_python/blob/master/machine_learning/regression/carprice_E1SUl6b.xlsx?raw=true', sheet_name='test')

##########데이터 분석

##########데이터 전처리

x_train_df = train_df.drop(['가격'], axis=1)
x_test_df = test_df.drop(['가격'], axis=1)
y_train_df = train_df['가격']
y_test_df = test_df['가격']

print(x_train_df.head())
'''
     년식  종류    연비   마력    토크   연료  하이브리드   배기량    중량 변속기
0  2015  대형   6.8  159  23.0  LPG      0  2359  1935  수동
1  2012  소형  13.3  108  13.9  가솔린      0  1396  1035  자동
2  2015  중형  14.4  184  41.0   디젤      0  1995  1792  자동
3  2015  대형  10.9  175  46.0   디젤      0  2497  2210  수동
4  2015  대형   6.4  159  23.0  LPG      0  2359  1935  자동
'''
print(x_train_df.columns) #Index(['년식', '종류', '연비', '마력', '토크', '연료', '하이브리드', '배기량', '중량', '변속기'], dtype='object')

transformer = make_column_transformer(
    (OneHotEncoder(), ['종류', '연료', '변속기']),
    remainder='passthrough')
transformer.fit(x_train_df)
x_train = transformer.transform(x_train_df) #트랜스포머의 transform() 함수는 결과를 넘파이 배열로 리턴
x_test = transformer.transform(x_test_df)

y_train = y_train_df.to_numpy()
y_test = y_test_df.to_numpy()

##########모델 생성

model = LinearRegression()

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_test, y_test)) #0.7739730315245023

##########모델 예측

x_test = [
    [2016, '대형', 6.8, 159, 25, 'LPG', 0, 2359, 1935, '수동']
]

x_test = transformer.transform(pd.DataFrame(x_test, columns=['년식', '종류', '연비', '마력', '토크', '연료', '하이브리드', '배기량', '중량', '변속기']))

y_predict = model.predict(x_test)
print(y_predict[0]) #1802.160302088625