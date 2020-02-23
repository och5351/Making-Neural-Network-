import numpy as np
from ChanHaeFlow.MyUtil import MyUtil
from ChanHaeFlow.Calculator import Calculator
from ChanHaeFlow.Neural import Neural
from tensorflow.keras.datasets import mnist

#난수 고정 재연
np.random.seed(100)

#데이터 셋 불러오기(학습 데이터 : 60,000 // 검증 데이터 : 10,000)
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#학습 및 검증 데이터 Flatten
X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32') / 255

#유틸 클래스
util = MyUtil()
neural = Neural()
calc = Calculator()

#원 핫 인코딩
Y_train = util.to_Catgorical(Y_train,10)
Y_test = util.to_Catgorical(Y_test, 10)

epoch = 2
learning_rate = 0.01

#데이터 입력
neural.data_holder(X_train, Y_train, epoch, learning_rate)

#입력층
print(neural.input_Layer)

#입력층 + 은닉층
neural.layer(20, 28*28, "Sigmoid")

#은닉층 + 출력층
neural.layer(10, len(neural.layer_predict[neural.get_Count()-1]),"Sigmoid")

#오차역전파 및 반복
neural.nn_compile("CrossEntropy")
