# coding: utf-8

import random
import copy
import threading
import numpy as np
from Cal.MatrixCreator import MatrixCreator
from Cal.GradientCal import GradientCal
from UsingTkinter import UsingTkinter

# 환경 변수 지정
random.seed(0)
np.random.seed(0)
# 입력값 및 타겟값
data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

# 실행 횟수(iterations), 학습률(lr), 모멘텀 계수(mo) 설정
iterations = 5000
lr = 0.1
mo = 0.4
mo2 = 0.999
e = 0.0000001
'''
알고리즘 설명
 * Layer는 해시 자료구조를 채택.
    ->  해시의 조회 속도는 배열의 조회 속도보다 빠르며 키를 Layer로 등록하여
       정렬 알고리즘을 사용하지 않고 각층 순전파, 역전파에서 조회가 가능하다(반복문에서 감소 반복으로 진행).
       PS. 정렬 알고리즘을 사용하지 않는 조건에서는 해시와 배열의 시간 복잡도는 의미가 없을 것 같다.
    ->  직관적인 Layer 확인이 가능하다.
 * 각 Layer 마다의 Activation, Weight 등은 배열인 list 채택.
    -> 완전 탐색 알고리즘을 요하기 때문에 시간 복잡도의 차이가 해시와 다르지 않다.

프로그램 구조 설명
 * Main 함수가 존재하는 Neural_Network.py.
 * 계산을 도와주는 Cal package의 GradientCal.py 와 MatrixCreator.py.
    ->  GradientCal.py : 미분과 활성화 함수 계산.
    ->  MatrixCreator.py : 모든 활성화, 가중치, 바이어스, 델타의 행렬을 제작.      
'''

# 신경망의 실행
class NeuralNetwork:

    # 초깃값의 지정
    def __init__(self, num_x, num_yh, num_yo):

        # Matrix 생성 클래스 호출
        self.m = MatrixCreator()
        self.g = GradientCal()
        # 입력값(num_x), 은닉층 초깃값(num_yh), 출력층 초깃값(num_yo)
        self.num_x = num_x
        self.num_yh = num_yh
        self.num_yo = num_yo

        t = threading.Thread(target=UsingTkinter,  # 쓰레드에 들어간 Tkinter Class
                             args=('Neural Network Structure', 1500, 800, self.num_x, self.num_yh, self.num_yo))
        t.start()

        # 활성화 함수 초깃값
        self.activationDic = self.m.makeActivation(self.num_x, self.num_yh, self.num_yo)
        # 가중치 초깃값
        self.weight = self.m.makeWeight(self.num_x, self.num_yh, self.num_yo)
        # 바이어스 초깃값
        # self.bias = self.m.makeBias(self.num_yh, self.num_yo)
        # 모멘텀 SGD를 위한 이전 가중치 초깃값 (깊은 복사로 객체 새로 생성)
        self.last_gradient = copy.deepcopy(self.weight)

    # 순전파 함수
    def propagate(self, inputs):

        # 입력층
        self.activationDic[0] = np.array(inputs, dtype=float)
        # 입력층 이외 노드 값 지정
        for layerCount in range(len(self.activationDic.keys())-1):
            sum = 0.0
            for nextNodeNum, _ in enumerate(self.activationDic[layerCount + 1]):  # 다음층 노드 순회
                for nodeNum, _ in enumerate(self.activationDic[layerCount]):  # 현재 층 노드 수회
                    sum += (self.activationDic[layerCount][nodeNum]   # 현재 층의 활성화 함수
                            * self.weight[layerCount][nodeNum][nextNodeNum])  # 현재 층의 노드와 다음 층 노드에 잇는 가중치
                             # + self.bias[layerCount][nextNodeNum]  # 다음 층 노드의 바이어스
                # 시그모이드와 tanh 중에서 활성화 함수 선택
                self.activationDic[layerCount + 1][nextNodeNum] = self.g.tanh(sum, False)

        return self.activationDic[len(self.activationDic)-1]

    # 역전파 함수
    def backPropagate(self, targets):

        # 델타 출력 계산
        deltas = self.m.makeDeltaMetrix(self.num_yh, self.num_yo)
        maxN = len(self.activationDic.keys()) - 1

        for layerCount in range(maxN, 0, -1):  # 각 층 역순회(Reverse)
            error = 0.0
            if layerCount == maxN:
                for nodeCount, _ in enumerate(self.activationDic[layerCount]):  # 현재 층 노드 순회
                    error = targets[nodeCount] - self.activationDic[layerCount][nodeCount]
                    # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
                    deltas[layerCount][nodeCount] = np.array([self.g.tanh(self.activationDic[layerCount][nodeCount], True) * error])
            else:
                for nodeDeltaCount, _ in enumerate(deltas[layerCount + 1]):  # 이전 델타 값 순회
                    for nodeCount, _ in enumerate(self.activationDic[layerCount]):  # 현재 층 노드 순회
                        error += deltas[layerCount + 1][nodeDeltaCount] * self.weight[layerCount][nodeCount][nodeDeltaCount]
                        # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
                        deltas[layerCount][nodeCount] = self.g.tanh(self.activationDic[layerCount][nodeCount], True) * error

        # 가중치 업데이트
        for layerCount in range(maxN, 0, -1):  # 각 층 역순회(Reverse)
            for preNodeCount, _ in enumerate(self.activationDic[layerCount - 1]):  # 이전 층 노드 순회
                for nodeCount, _ in enumerate(self.activationDic[layerCount]):  # 현재 층 노드 순회
                    gradient = deltas[layerCount][nodeCount] * self.activationDic[layerCount - 1][preNodeCount]
                    # Adam 업데이트
                    V = (mo * self.last_gradient[layerCount - 1][preNodeCount][nodeCount] + ((1 - mo) * gradient))
                    G = (mo2 * self.last_gradient[layerCount - 1][preNodeCount][nodeCount] + ((1 - mo2) * gradient ** 2))
                    V /= (1 - mo)
                    G /= (1 - mo2)
                    self.weight[layerCount-1][preNodeCount][nodeCount] += -lr*(G/np.sqrt(V + e))
                    # Momentum 업데이트
                    # v = mo * self.last_gradient[layerCount - 1][preNodeCount][nodeCount] - lr * gradient
                    # self.weight[layerCount - 1][preNodeCount][nodeCount] += v
                    self.last_gradient[layerCount-1][preNodeCount][nodeCount] = gradient

        # 오차의 계산(최소 제곱법)
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.activationDic[maxN][k]) ** 2
        return error

    # 학습 실행
    def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.propagate(inputs)
                error = error + self.backPropagate(targets)
            if i % 500 == 0:
                print('error: %-.5f' % error)

    # 결괏값 출력
    def result(self, patterns):
        for p in patterns:
            print('Input: %s, Predict: %s' % (p[0], self.propagate(p[0])))


if __name__ == '__main__':
    # 두 개의 입력 값, 두 개의 레이어, 하나의 출력 값을 갖도록 설정
    n = NeuralNetwork(2, [3,2], 1)

    # 학습 실행
    n.train(data)

    # 결괏값 출력
    n.result(data)
