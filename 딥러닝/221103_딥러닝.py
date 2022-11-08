import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print(tf.__version__)

import keras
print(keras.__version__)

# 케라스의 인공 신경망 모델 만들기
    # 1. keras.models 모듈의 Sequential객체를 통한 순차 모델을 사용하는 방법
        #   - layer를 순차적으로 추가해나가는 방법임
        #   - 간단하고 쉽게 모델을 만들 수 있음
    # 2. keras.models 모듈의 Model객체를 통한 functional API를 사용하여 만드는 방법
        #   - 훨씬 다양한 모델을 만들 수 있음

model = keras.models.Sequential()
print(type(model))

# 케라스의 층(layer)에는 여러 종류가 있는데,
# 지금은 이전 layer와 현재 layer의 모든 유닛이 다 연결되어 있는 keras.layers모듈의 Dense layer를 사용할 거임
# keras.layer.Dense(유닛의수, 입력자료의 차원모양)

# #### 모델을 구축 및 학습하는 프로세스
    # 1. 모델 학습과정 설정
        #   - 손실함수 및 최적화 방법을 정의함
        #   - keras에서는 compile()함수 사용함
    # 2. 모델 학습시키기
     #   - 훈련셋 & fit() 사용
    # 3. 학습 과정 살펴보기
        #   - 모델 학습 시 훈련셋과 검증셋의 손실 및 정확도 측정하기
        #   - 반복횟수에 따른 손실 및 정확도 추이를 보면서 학습상황 판단
    # 4. 모델 평가
        #   - 테스트 셋으로 평가 & evaluate()사용
    # 5. 모델 사용
        #   - 새로운 데이터 입력 & predict() 사용

# ### 모델의 종류
# - Sequential모델
# - functional모델

#### 1.Sequential모델
#### Dense Layer를 이용한 Sequential모델 구축

# ![image](https://user-images.githubusercontent.com/87592790/199675628-33cf09df-c704-4ec4-95c4-dbe86c5bcbe7.png)

# - Dense layer란? 입력과 출력이 모두 연결된 layer/ 노드로 구성된 리스트
# - 위 그림을 통해 아래 코드를 이해해보자
# - 우리는 총 3개의 layer를 추가하는 코드를 작성했다.(3개인가 4개인가?)

# 선형회귀모델의 hypotheses와 cost
# H(x) = Wx + b
# Cost(W,b) = (1/m) (m시그마i=1)(H(x^i) - y^i)^2
# 우리는 선형회귀 모델을 사용해서 학습을 시킬건데, 
# 입력데이터 x와 출력레이블 y를 주고 학습하는데 이것 과정은 W와 b를 구하는 과정이라고 할 수 있다.

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import numpy as np 

# 1.데이터 준비
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
x_pred = np.array([6,7,8,9,10])

# 2.모델 구축
    # ? 활성함수란
    
# sequential모델 구축 방법1
    # 층을 만들어가는 과정 : add()는 레이어를 추가해줌 / Dense(출력노드 갯수, input_dim(입력노드 갯수), activation(활성화함수))
# model = Sequential() 
# model.add(Dense(units=5, input_dim = 1, activation = 'relu'))
# model.add(Dense(units=3)) # 출력노드가 3개
# model.add(Dense(units=1)) # 출력노드가 1개

# sequential모델 구축 방법2
model = keras.Sequential([
    keras.layers.Dense(units=5, input_dim = 1, activation = 'relu'),
    keras.layers.Dense(units=3),
    keras.layers.Dense(units=1)
])

# 모델 확인
model.summary() # ??
    # 트레이닝 데이터 여러 개를 동시에 학습을 시키기 때문에 Output Shape의 차원 수는 input_shape 으로 지정한 것보다 하나가 더 많고
    # 몇 개를 동시에 학습시킬 것인지는 현재 알 수 없으므로 None으로 출력된다 

# 이미지로 확인
plot_model(model, show_shapes=True, dpi=80 ) # 왜 안되는거얌...--

# 3.모델 준비
    # 손실함수, 최적화방법, 평가지표
    # model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='mse', optimizer='sgd')

# 4.모델 학습
model.fit(x, y, epochs=100, batch_size=1, verbose=0) 
    # epoch : x데이터를 100번 사용해서 훈련
    # batchsize :  한 번 훈련할 때 나누는 범위
    # verbose = 0 :  학습하는 동안 중간에 log를 출력하지 않는다는 뜻
    # loss의 값을 보면 점점 줄어드는 것을 알 수 있는데, 맞추는 확률이 늘고있다는 의미이다.
    
# 5.모델 평가
print("평가값")
model.evaluate(x,y, batch_size=1)

# 6.예측
print("예측값")
model.predict(x_pred)
# [6,7,8,9,10]을 넣었을 때 값이 비슷하게 나오지만 손실함수가 0이 아니라 약간의 오차가 발생함을 알 수 있다.


# ## 2.Functional(함수형) 모델!
# #### 단순 functional 모델 : functional API사용해 구축
    # - 특징
        # - sequential모델에 비해 더 자유롭게 만들 수 있다.
        # - 입력크기(shape)를 명시한 입력층을 앞에서 정의해줘야 한다.

# 1.데이터 준비
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
x_pred = np.array([6,7,8,9,10])

# 2.모델 구축
    # 입력층, 은닉층, 출력층 정의하기
input = keras.layers.Input( shape=(1,) )  # 입력노드 생성
hidden1 = keras.layers.Dense(5, activation='relu')(input)  # 다음 노드를 생성 f(x) = h라고 생각하면 됨
hidden2 = keras.layers.Dense(3, activation='relu')(hidden1)  # 다음 노드를 생성 f(x) = h라고 생각하면 됨
output = keras.layers.Dense(1)(hidden2)  # g(h) = y
 
# 참고
    # print(type(input)) #<class 'keras.engine.keras_tensor.KerasTensor'>로 아직 유닛과 연결되지 않은 상태임을 알 수 있다.
    # print(type(keras.layers.Dense(2))) #<class 'keras.layers.core.dense.Dense'>로 layer과 유닛이 연결된 상태이다.
    # Dense layer가 된 keras.layers.Dense(2)를 d라고 한다면
    # y = d(x)처럼 layer객체를 함수처럼 사용할 수 있다.(Dense layer는 call메소드가 정의되어 있는 callable객체임)
    # 모델에 입력,출력을 정의해줌
model = keras.models.Model(input, output) 
# model.summary()

print("모델 학습 전")
y_predict = model.predict( x ) 
print( y_predict.flatten() ) # [0.03701413 0.07402827 0.1110424  0.14805654 0.18507066]
print(y, '\n')
# 결과를 보면 거의 맞추지 못했다는 것을 알 수 있다.

model.compile( 'SGD', 'mse' ) 
model.fit( x, y, epochs = 1000, verbose = 0 )
 
print("모델 학습 후")
y_predict = model.predict( x ) 
print( y_predict.flatten() ) # [1.0000356 2.0000222 3.0000088 3.9999952 4.999982 ]
print( y )
# compile과 fit을 거쳐 오차를 줄여갔다.

# #### 3.Class로 이루어진 모델
