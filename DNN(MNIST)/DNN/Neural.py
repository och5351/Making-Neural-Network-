import numpy as np
from DNN.Calculator import Calculator as calc

class Neural:
    # 파라메타
    __eta = 0
    __epoch = 0
    __batch = 0
    # 사용한 층
    __count = 1
    # 데이터 받아오기
    __X_train = []
    __Y_train = []
    # 테스트 데이터 받아오기
    __X_test = []
    __Y_test = []
    # 입력층
    input_Layer = []
    # 이전 층 예측 값들
    layer_predict = []
    # 모든 가중치(InputLayer = 0 ~ OutputLayer = count-1)
    w = []
    # 모든 바이어스(InputLayer = 0 ~ OutputLayer = count-1)
    b = []
    # 현재 진행률 분자 변수
    __nowWhere = 0
    # 검증 스위치
    __testOn = 0
    #Layer 구성 값 : [[노드 개수, 입력 데이터 개수, 활성화 함수]]
    __layerParameter = []

    def __init__(self):
        print("Neural Class Call")

    # 이전층 예측값 수 또는 입력층 데이터 수 출력
    def getPreLayerOutPut(self):
        return self.__layerParameter[self.__count - 1][1]

    # 데이터 담아두기
    def data_holder(self, inputData, inputY_Class, Xtest, Ytest, epoch, batch, eta):
        self.__X_train = inputData #학습데이터 저장
        self.input_Layer = (self.__X_train[0])  # 첫 입력층
        self.__Y_train = inputY_Class #학습 정답 데이터 저장
        self.__X_test = Xtest #검증 데이터 저장
        self.__Y_test = Ytest #검증 정답 데이터 저장
        self.__layerParameter.append([len(self.__X_train[0]),len(self.__X_train[0]),0])
        self.__epoch = epoch
        self.__batch = batch
        self.__eta = eta


    # 가중치 초기화(Layer당 한번씩)
    def initWeight(self, x):
        tempW = np.random.normal(0, 1, x)
        tempW = np.expand_dims(tempW, axis=0)
        return tempW

    # 편향 초기화
    def initBias(self, x):
        tempb = np.zeros(x)
        # 바이어스가 처음 초기화 될 때
        self.b.append(tempb)

    # 1차원 배열로 이용
    def unit(self, x, w, b):
        return np.sum(x * w) + b

    # 노드 함수
    def node(self, unit, activation):
        if activation == "SIGMOID":
            return calc.sigmoid(calc, unit)
        elif activation == "RELU":
            return calc.ReLu(calc, unit)

    def layer(self, nodeCount, inputDataCount, activation):
        temp_w = []

        # 가중치 초기화
        for nc in range(nodeCount):
            if (nc > 0):
                temp_w = np.append(temp_w, self.initWeight(self.__layerParameter[self.__count - 1][0]), axis=0)
            else:
                temp_w = self.initWeight(self.__layerParameter[self.__count - 1][0])

        # 바이어스 초기화
        self.initBias(nodeCount)

        self.w.append(temp_w)
        tempLayerParameter = [nodeCount, inputDataCount, activation] # 사용할 층 정보 저장
        self.__layerParameter.append(tempLayerParameter) # 정보 리스트화

        self.__count += 1  # 사용 층 증가

    def forwardPropagation(self):
        # 층 개수
        for layerCount in range(self.__count - 1):
            temp_predict = []
            # 입력 데이터를 옮길 때
            if layerCount == 0:
                # 다음 층 노드 개수
                for nextNodeCount in range(self.__layerParameter[layerCount + 1][0]):
                    # 다음 층 개별 유닛 값
                    unit = self.unit(self.input_Layer, self.w[layerCount][nextNodeCount],
                                     self.b[layerCount][nextNodeCount])
                    # 노드 예측값 임시 저장
                    temp_predict = np.append(temp_predict, self.node(unit,
                                                                     self.__layerParameter[layerCount + 1][2].upper()))
                if self.__nowWhere == 0:
                    self.layer_predict.append(temp_predict)
                else:
                    self.layer_predict[layerCount] = temp_predict
            # 입력층 이후 순전파
            else:
                # 다음 층 노드 개수
                for nextNodeCount in range(self.__layerParameter[layerCount + 1][0]):
                    # 다음 층 개별 유닛 값
                    unit = self.unit(self.layer_predict[layerCount-1], self.w[layerCount][nextNodeCount],
                                     self.b[layerCount][nextNodeCount])
                    # 노드 예측값 임시 저장
                    temp_predict = np.append(temp_predict, self.node(unit,
                                                                     self.__layerParameter[layerCount + 1][2].upper()))
                if self.__nowWhere == 0 and self.__testOn == 0:
                    self.layer_predict.append(temp_predict)
                else:
                    self.layer_predict[layerCount] = temp_predict

    def backPropagation(self):
        # 치환 변수
        delta = [0, 0]

        # 층 개수
        for layerCount in reversed(range(self.__count - 1)):
            # 출력 층
            if layerCount == self.__count - 2:
                # 이전 층 노드 개수
                for preNodeCount in range(self.__layerParameter[layerCount][0]):
                    # 현재 층 노드 개수
                    for currNodeCount in range(self.__layerParameter[layerCount + 1][0]):
                        w_diff, b_diff, temp_delta = \
                            calc.sigmoid_OutLayer_Gradient_Descent(calc, self.layer_predict[layerCount][currNodeCount],
                                                                   self.__Y_train[self.__nowWhere][currNodeCount],
                                                                   self.layer_predict[layerCount - 1][preNodeCount],
                                                                   self.w[layerCount][currNodeCount][preNodeCount])
                        # 출력층 가중치 업데이트
                        self.w[layerCount][currNodeCount][preNodeCount] -= (self.__eta * w_diff)

                        if preNodeCount + 1 == self.__layerParameter[layerCount][0]:
                            # 편향 업데이트
                            self.b[layerCount][currNodeCount] -= (self.__eta * b_diff)
                            # 이전 층을 위한 delta 치환
                            delta[0] += temp_delta
            # 마지막 역전파
            elif layerCount == 0:
                # 이전 층 노드 개수
                for preNodeCount in range(self.__layerParameter[layerCount][0]):
                    # 현재 층 노드 개수
                    for currNodeCount in range(self.__layerParameter[layerCount + 1][0]):
                        w_diff, b_diff, temp_delta = \
                            calc.sigmoid_HiddenLayer_Gradient_Descent(calc,
                                                                      self.layer_predict[layerCount][currNodeCount],
                                                                   delta[0],
                                                                   self.input_Layer[preNodeCount],
                                                                   self.w[layerCount][currNodeCount][preNodeCount])
                        # 입력층 가중치 업데이트
                        self.w[layerCount][currNodeCount][preNodeCount] -= (self.__eta * w_diff)
                        if preNodeCount + 1 == self.__layerParameter[layerCount][0]:
                            # 편향 업데이트
                            self.b[layerCount][currNodeCount] -= (self.__eta * b_diff)
            # 은닉층
            else:
                # 이전 층 노드 개수
                for preNodeCount in range(self.__layerParameter[layerCount][0]):
                    # 현재 층 노드 개수
                    for currNodeCount in range(self.__layerParameter[layerCount + 1][0]):
                        w_diff, b_diff, temp_delta = \
                            calc.sigmoid_HiddenLayer_Gradient_Descent(calc,
                                                                      self.layer_predict[layerCount][currNodeCount],
                                                                      delta[0],
                                                                      self.input_Layer[preNodeCount],
                                                                      self.w[layerCount][currNodeCount][preNodeCount])
                        # 입력층 가중치 업데이트
                        self.w[layerCount][currNodeCount][preNodeCount] -= (self.__eta * w_diff)

                        if preNodeCount + 1 == self.__layerParameter[layerCount][0]:
                            # 편향 업데이트
                            self.b[layerCount][currNodeCount] -= (self.__eta * b_diff)
                            # 이전 층을 위한 delta 치환
                            delta[1] += temp_delta
                    if preNodeCount + 1 == self.__layerParameter[layerCount][0]:
                        # 델타 값 업데이트
                        delta[0] = delta[1]
                        delta[1] = 0

        # nn 통과 횟수 증가
        self.__nowWhere += 1


    def runNN(self, loss):
        # 학습 횟수
        for learning in range(20001):
            # 순 전파
            self.forwardPropagation()
            # 오차 역 전파
            self.backPropagation()
            # 손실
            if learning % 100 == 0:
                if loss.upper() == "CROSSENTROPY":
                    print("진행도 : {son}%".format(son=round(self.__nowWhere / 20000, 3)*100))
                    print("[손실]")
                    print(calc.crossEntropy(calc, self.layer_predict[1], self.__Y_train[self.__nowWhere - 1]))
                    print("[예측값]")
                    print(self.layer_predict[1])
                    print("[정답]")
                    print(self.__Y_train[self.__nowWhere - 1])
                    print("="*60)
            # 데이터 교체
            self.input_Layer = self.__X_train[self.__nowWhere]
        print("="*60)
        print("\n\t검증\n")
        print("=" * 60)
        self.__nowWhere = 0
        for learning in range(5):
            self.__testOn = 1
            self.input_Layer = self.__X_test[self.__nowWhere]
            self.__Y_train = self.__Y_test
            # 순 전파
            self.forwardPropagation()
            # 오차 역 전파
            #self.backPropagation()
            # 손실
            if loss.upper() == "CROSSENTROPY":
                print("[손실]")
                print(calc.crossEntropy(calc, self.layer_predict[1], self.__Y_train[self.__nowWhere - 1]))
                print("[예측값]")
                print(self.layer_predict[1])
                print("[정답]")
                print(self.__Y_train[self.__nowWhere - 1])
                print("=" * 60)
            self.__nowWhere += 1
            # 데이터 교체