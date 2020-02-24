import numpy as np

class Calculator:

    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))

    def ReLu(self, x):
        if x < 0:
            x = 0.0
        return x

    def crossEntropy(self, y_predict, y_class):
        return -np.sum(y_class * np.log(y_predict) + (1 - y_class) * np.log(1 - y_predict))

    # 시그모이드 경사하강법
    '''
        * Gradient Descent 출력층 일반화
    '''
    def sigmoid_OutLayer_Gradient_Descent(self, Y_predict, Y_target, pre_Y_predict, w2):
        wdiff = (Y_predict - Y_target)*Y_predict*(1-Y_predict)*pre_Y_predict
        bdiff = (Y_predict - Y_target)*Y_predict*(1-Y_predict)
        delta = (Y_predict - Y_target)*Y_predict*(1-Y_predict)*w2
        return wdiff, bdiff, delta
    '''
        * Gradient Descent 은닉층 일반화
    '''
    def sigmoid_HiddenLayer_Gradient_Descent(self, Y_predict, i_delta, pre_Y_predict, w2):
        wdiff = i_delta * Y_predict * (1 - Y_predict)*pre_Y_predict
        bdiff = i_delta * Y_predict * (1 - Y_predict)
        delta = i_delta * Y_predict * (1 - Y_predict) * w2
        return wdiff, bdiff, delta

    # ReLu 경사하강법
    def ReLu_OutLayer_Gradient_Descent(self,Y_predict,Y_target,pre_Y_predict):
        return (Y_predict - Y_target) * Y_predict * pre_Y_predict

