import numpy as np
from ChanHaeFlow.Calculator import Calculator as calc

class Neural:

    #파라메타
    __eta = 0
    __epoch = 0
    #사용한 층
    __count = 0
    #데이터 받아오기
    __X_data = []
    __Y_Class = []
    #입력층
    input_Layer = []
    # 이전 층 예측 값들
    layer_predict = []
    #모든 가중치(InputLayer = 0 ~ OutputLayer = count-1)
    w = []
    #모든 바이어스(InputLayer = 0 ~ OutputLayer = count-1)
    b = []
    #현재 진행률 분자 변수
    __nowWhere = 0

    #predict index
    def get_Count(self):
        return self.__count - 1

    def getY(self):
        return self.__Y_Class

    #데이터 담아두기
    def data_holder(self, inputData, inputY_Class, epoch, eta):
        self.__X_data = inputData
        self.__Y_Class = inputY_Class
        self.__epoch = epoch
        self.__eta = eta
        self.input_Layer = (self.__X_data[0]) #첫 입력층

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


    #layer 함수(노드 수, 인풋 데이터 수, 활성화 함수 종류 )
    def layer(self, nodeCount,  inputDataCount, activation):
        if self.__nowWhere == 0:
            self.initBias(nodeCount) #바이어스 초기화
            tempW = 0
            temp_predict = []

        for i in range(nodeCount):
            if self.__nowWhere == 0:
                if(i > 0):
                    tempW = np.append(tempW, self.initWeight(inputDataCount), axis=0)
                else:
                    tempW = self.initWeight(inputDataCount)

                if self.__count == 0:
                    tempX = self.unit(self.__X_data[self.get_Count()], tempW[i], self.b[self.__count][i])
                else:
                    tempX = self.unit(self.layer_predict[self.__count-1], tempW[i], self.b[self.__count][i])

                temp_predict = np.append(temp_predict,self.node(tempX, activation))
            else:
                if self.__count == 0:
                    tempX = self.unit(self.__X_data[self.get_Count()], self.w[self.__count][i], self.b[self.__count][i])

                else:
                    tempX = self.unit(self.layer_predict[self.__count - 1], self.w[self.__count][i], self.b[self.__count][i])
                self.layer_predict[self.__count][i] = self.node(tempX, activation)


        if self.__nowWhere == 0:
            self.layer_predict.append(temp_predict)
            #한층을 지난 가중치 저장
            self.w.append(tempW)
            #층 갯수 증가
        self.__count = self.__count + 1

    def back_Propagation(self, resultCount = 0):
        self.__nowWhere += 1
        pre_Layer_Predict_Count = 0

        delta = [0.0, 0.0]
        for layerCount in reversed(range(self.__count)):
            for nodeCount in range(len(self.w[layerCount])):
                for weightCount in range(len(self.w[layerCount][nodeCount])):
                    if layerCount != 0:

                        if pre_Layer_Predict_Count == len(self.layer_predict[layerCount - 1]):
                            pre_Layer_Predict_Count = 0

                        if layerCount == (self.__count-1): #출력층 가중치 업데이트

                            #출력층 경사하강  @@@(전체 for문의 인수로 __Y_Class[0]의 하드코딩 바꿔주기!!)@@@
                            result = calc.sigmoid_OutLayer_Gradient_Descent(calc, self.layer_predict[layerCount][nodeCount], self.__Y_Class[resultCount][nodeCount],
                                                                           self.layer_predict[layerCount-1][pre_Layer_Predict_Count])
                            #은닉층 -> 출력층 가중치 업데이트
                            self.w[layerCount][nodeCount][weightCount] = self.w[layerCount][nodeCount][weightCount] - (self.__eta*result[0])

                            delta[0] += result[1]    #델타 값 미리 더해 놓기
                        else:
                            result = calc.sigmoid_HiddenLayer_Gradient_Descent(calc, self.layer_predict[layerCount][nodeCount], delta[0],
                                                                              self.layer_predict[layerCount-1][pre_Layer_Predict_Count])

                            self.w[layerCount][nodeCount][weightCount] = self.w[layerCount][nodeCount][weightCount] - (self.__eta * result[0])
                            delta[1] += result[1]  #델타 값 미리 더해 놓기
                            if (nodeCount + 1) == len(self.w[layerCount]):  #델타 갱신
                                delta[0] = delta[1]
                                delta[1] = 0

                    else:

                        if pre_Layer_Predict_Count == len(self.input_Layer):
                            pre_Layer_Predict_Count = 0
                        result = calc.sigmoid_HiddenLayer_Gradient_Descent(calc, self.layer_predict[layerCount][nodeCount],  delta[0],
                                                                          self.input_Layer[pre_Layer_Predict_Count])

                        self.w[layerCount][nodeCount][weightCount] = self.w[layerCount][nodeCount][weightCount] - (self.__eta * result[0])

                    pre_Layer_Predict_Count += 1
        self.__count = 0



    def nn_compile(self,loss):


        # 손실 계산
        if loss == "CrossEntropy":
            loss = calc.crossEntropy(calc, self.layer_predict[1], self.__Y_Class[0])
        elif loss == "MSE":
            loss = calc.crossEntropy(calc, self.layer_predict[1], self.__Y_Class[0])

        print("[손실]")
        print(loss)
        print("=" * 50)
        self.back_Propagation()
        for learning in range(len(self.__X_data)-1):
            realNum = learning + 1
            self.input_Layer = self.__X_data[realNum]
            self.layer(len(self.layer_predict[0]), len(self.input_Layer), "Sigmoid")
            self.layer(len(self.layer_predict[1]), len(self.layer_predict[0]), "Sigmoid")
            self.back_Propagation()
            if realNum % 1000 == 0:
                loss = calc.crossEntropy(calc, self.layer_predict[1], self.__Y_Class[realNum])
                print("[손실]")
                print(loss)
                print("=" * 50)
                if loss < 1:
                    break

