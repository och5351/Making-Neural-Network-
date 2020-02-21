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

epoch = 2001

#데이터 입력
neural.data_holder(X_train, Y_train)

#입력층 + 은닉층
neural.layer(20,28*28,"Sigmoid")
print("[입력층 -> 은닉층 1]")
print(neural.layer_predict[neural.get_Count()])

print("\n")

#은닉층 + 출력층
print("[은닉층 -> 출력층]")
neural.layer(10,len(neural.layer_predict[neural.get_Count()-1]),"Sigmoid")
print(neural.layer_predict[neural.get_Count()])

loss = calc.crossEntropy(neural.layer_predict[1],Y_train[0])
print("\n")
print("[손실]")
print(loss)

neural.back_Propagation()