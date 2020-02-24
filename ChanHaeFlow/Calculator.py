import numpy as np

class Calculator:

    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))

    def ReLu(self, x):
        if x < 0:
            x = 0.0
        return x

    def to_Catgorical(self, x, i=0):
        if i == 0:
            i = x
        result = np.arange(i)
        for j in range(i):
            if j == x - 1:
                result[j] = 1
            else:
                result[j] = 0
        return result

    def crossEntropy(self, y_predict, y_class):
        return -np.sum(y_class * np.log(y_predict) + (1 - y_class) * np.log(1 - y_predict))

    '''
        * Gradient Descent 출력층 일반화
    '''
    # 시그모이드 경사하강법
    def sigmoid_OutLayer_Gradient_Descent(self,Y_predict,Y_target,pre_Y_predict, w2):
        wdiff = (Y_predict - Y_target)*Y_predict*(1-Y_predict)*pre_Y_predict
        bdiff = (Y_predict - Y_target)*Y_predict*(1-Y_predict)
        wdelta = (Y_predict - Y_target)*Y_predict*(1-Y_predict)*w2
        bdelta = (Y_predict - Y_target)*Y_predict*(1-Y_predict)*w2
        result = [wdiff, wdelta, bdiff, bdelta]
        return result

    # ReLu 경사하강법
    def ReLu_OutLayer_Gradient_Descent(self,Y_predict,Y_target,pre_Y_predict):
        return (Y_predict - Y_target) * Y_predict * pre_Y_predict

    '''
        * Gradient Descent 은닉층 일반화
    '''
    def sigmoid_HiddenLayer_Gradient_Descent(self,Y_predict, pre_delta, pre_Y_predict, pre_bdelta):
        wdiff = pre_delta*Y_predict*(1-Y_predict)*pre_Y_predict
        bdiff = pre_bdelta*Y_predict*(1-Y_predict)
        delta = pre_delta*Y_predict*(1-Y_predict)*Y_predict
        result = [wdiff, delta, bdiff]
        return result

