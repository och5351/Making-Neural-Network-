import numpy as np
from ChanHaeFlow.Calculator import Calculator as calc

class Neural:

    #사용한 층
    __count = 0
    #데이터 받아오기
    __X_data = []
    __Y_Class = []
    # 이전 층 예측 값들
    layer_predict = []
    #모든 가중치(InputLayer = 0 ~ OutputLayer = count-1)
    w = []
    #모든 바이어스(InputLayer = 0 ~ OutputLayer = count-1)
    b = []

    #predict index
    def get_Count(self):
        return self.__count - 1

    #데이터 담아두기
    def data_holder(self, inputData, inputY_Class):
        self.__X_data = inputData
        self.__Y_data = inputY_Class

    #가중치 초기화(Layer당 한번씩)
    def initWeight(self,x):
        tempW = np.random.normal(0, 1, x)
        tempW = np.expand_dims(tempW, axis=0)
        return tempW

    #편향 초기화
    def initBias(self, x):
        tempb = np.zeros(x)
        #바이어스가 처음 초기화 될 때
        self.b.append(tempb)

    # 1차원 배열로 이용
    def unit(self, x, w, b):
        return np.sum(x * w) + b

    #노드 함수
    def node(self, unit, activation):
        if activation == "Sigmoid":
            return calc.sigmoid(calc, unit)
        elif activation == "ReLu":
            return calc.ReLu(calc, unit)


    #layer 함수(노드 수, 인풋 데이터 수, 활성화함수 종류 )
    def layer(self, nodeCount,  inputDataCount, activation):
        self.initBias(nodeCount) #바이어스 초기화
        tempW = 0
        temp_predict = []

        for i in range(nodeCount):
            if(i > 0):
                tempW = np.append(tempW, self.initWeight(inputDataCount), axis=0)
            else:
                tempW = self.initWeight(inputDataCount)

            if self.__count == 0:
                tempX = self.unit(self.__X_data[self.get_Count()], tempW[i], self.b[self.__count][i])
            else:
                tempX = self.unit(self.layer_predict[self.__count-1][i], tempW[i], self.b[self.__count][i])

            temp_predict = np.append(temp_predict,self.node(tempX, activation))

        self.layer_predict.append(temp_predict)
        #한층을 지난 가중치 저장
        self.w.append(tempW)
        #층 갯수 증가
        self.__count = self.__count + 1

    def back_Propagation(self):
        for layerCount in reversed(range(self.__count)):
            for nodeCount in range(len(self.w[layerCount])):
                for weightCount in range(len(self.w[layerCount][nodeCount])):
                    if layerCount == self.__count: #출력층 가중치 업데이트
                        #출력층 경사하강
                        wdiff = calc.sigmoid_Gradient_Descent(calc, self.layer_predict[layerCount][nodeCount],self.__Y_Class[nodeCount],self.layer_predict[layerCount-1][nodeCount-1])
                        #은닉층 -> 출력층 가중치 업데이트
                        self.w[layerCount][nodeCount][weightCount] = self.w[layerCount][nodeCount][weightCount] - wdiff
                        #
                    else:

                    #print(self.w[layerCount][nodeCount][weightCount])