import numpy as np
from DNN.MyUtil import MyUtil
from DNN.Calculator import Calculator
from DNN.Neural import Neural
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
Y_train = util.to_Catgorical(Y_train, 10)
Y_test = util.to_Catgorical(Y_test, 10)

epoch = 2001
batch = 10
learning_rate = 0.001

#데이터 입력(입력층 자동 생성)
neural.data_holder(X_train, Y_train, X_test, Y_test, epoch, batch, learning_rate)

#은닉층1
neural.layer(20, 28*28, "Sigmoid")

#출력층
neural.layer(10, neural.getPreLayerOutPut(), "Sigmoid")

neural.runNN("crossentropy")
