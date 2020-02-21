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

    def sigmoid_Gradient_Descent(self,Y_predict,Y_target,pre_Y_predict):
        return (Y_predict - Y_target)*Y_predict*(1-Y_predict)*pre_Y_predict

'''
    def MSE(self, y_predict, y_class):
        return np.sum((y_predict - y_class) ** 2) / x1.size

'''
