# -*- coding: utf-8 -*-

import random
import numpy as np
import copy

random.seed(777)

# 환경 변수 지정

# 입력값 및 타겟값
data = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]

# 실행 횟수(iterations), 학습률(lr), 모멘텀 계수(mo) 설정
iterations = 10000
lr = 0.1
mo = 0.4


# 활성화 함수 - 1. 시그모이드
# 미분할 때와 아닐 때의 각각의 값
def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# 활성화 함수 - 2. tanh
# tanh 함수의 미분은 1 - (활성화 함수 출력의 제곱)
def tanh(x, derivative=False):
    if derivative:
        return 1 - x ** 2
    return np.tanh(x)


# 가중치 배열 만드는 함수
def makeMatrix(i, j, fill=0.0):
    mat = []
    for i in range(i):
        mat.append([fill] * j)
    return mat

# 신경망의 실행
class NeuralNetwork:

    # 초깃값의 지정
    def __init__(self, num_x, num_yh, num_yo, bias=1):
        '''
            :param num_x: 입력 데이터(NeuralNetwork Class 는 num_x 안에 Bias가 끝부분에 생성 되므로 주의)
            :param num_yh: 은닉 층 노드 수 배열
            :param num_yo: 출력층 노드 수
            :param bias: 바이어스는 업데이트하는 코드가 없으므로 무시(매개변수로만 끝나는 변수)
        '''
        # 입력값(num_x), 은닉층 초깃값(num_yh), 출력층 초깃값(num_yo), 바이어스
        self.num_x = num_x  # 바이어스는 1로 지정(본문 참조)
        self.num_yh = num_yh  # 리스트로 변경
        self.num_yo = num_yo
        self.bias = bias
        temp2 = copy.deepcopy([self.num_x])
        temp2.extend(self.num_yh)
        temp2.append(self.num_yo)
        temp2.reverse()
        # 출력층에서 부터 입력층까지의 노드
        #print('-' * 200)
        #print('출력층에서 부터 입력층까지의 노드(Layer Structure)')
        self.LayerStructure = temp2
        #print(self.LayerStructure)
        # 활성화 함수 초깃값
        self.activation_input = [1.0] * self.num_x
        # 은닉층 활성화 함수 초깃값
        self.activation_hidden = []
        for nodeNum in self.num_yh:
            self.activation_hidden.append([1.0] * nodeNum)

        self.activation_out = [1.0] * self.num_yo
        # 가중치 초깃값
        self.weight_in = []
        temp = self.num_x
        for layerNumCount in range(len(self.num_yh)):
            self.weight_in.append(makeMatrix(temp, self.num_yh[layerNumCount]))
            temp = self.num_yh[layerNumCount]
        # 모멘텀 SGD를 위한 이전 가중치 초깃값
        self.gradient_in = copy.deepcopy(self.weight_in)
        # 가중치 초깃값 초기화
        temp = self.num_x
        for i in range(len(self.weight_in)):  # Layer Counter
            for j in range(temp):
                for k in range(len(self.weight_in[i][j])):
                    self.weight_in[i][j][k] = random.random()
            if i < len(self.weight_in)-1:
                temp = len(self.weight_in[i+1])
        #print('-' * 200)
        #print('가중치 초깃값 난수화')
        #print(self.weight_in)
        # 가중치 출력 초깃값
        self.weight_out = makeMatrix(self.num_yh[len(num_yh)-1], self.num_yo)
        # 모멘텀 SGD를 위한 이전 출력층 가중치 초깃값
        self.gradient_out = copy.deepcopy(self.weight_out)

        for j in range(self.num_yh[len(num_yh)-1]):
            for k in range(self.num_yo):
                self.weight_out[j][k] = random.random()
        #print('-' * 200)
        #print('출력층 가중치 난수화')
        #print(self.weight_out)



    # 업데이트 함수
    def update(self, inputs):
        '''
        :param inputs: 입력 데이터
        :return: 출력값
        '''

        # 입력 레이어의 활성화 함수
        for i in range(self.num_x):
            self.activation_input[i] = inputs[i]

        # 은닉층의 활성화 함수
        temp = self.activation_input
        for i in range(len(self.num_yh)):  # 은닉층 갯수
            for j in range(self.num_yh[i]):  # Layer 별 Node 갯수
                sum = 0.0
                for k in range(len(temp)):  # 이전 층 노드 갯수
                    sum += ((temp[k] * self.weight_in[i][k][j]) + self.bias)  # 유닛의 값 등록 (이전 노드들 -> 현재 노드)

                # 시그모이드와 tanh 중에서 활성화 함수 선택
                self.activation_hidden[i][j] = tanh(sum, False)

            temp = self.activation_hidden[i]
        #print('-' * 200)
        #print('은닉층 유닛의 값 출력')
        #print(self.activation_hidden)
        # 출력층의 활성화 함수
        for k in range(self.num_yo):  # Output Layer Node Count
            sum = 0.0
            for j in range(self.num_yh[len(self.num_yh)-1]):  # Last Hidden Layer Node Count
                sum = sum + ((self.activation_hidden[len(self.activation_hidden)-1][j] * self.weight_out[j][k]) + self.bias)

            # 시그모이드와 tanh 중에서 활성화 함수 선택
            self.activation_out[k] = tanh(sum, False)
        #print('-' * 200)
        #print('출력값')
        #print(self.activation_out[:])

        return self.activation_out[:]

    # 역전파의 실행
    def backPropagate(self, targets):

        # 델타 출력 계산
        output_deltas = [0.0] * self.num_yo
        for k in range(self.num_yo):
            error = targets[k] - self.activation_out[k]
            # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
            output_deltas[k] = tanh(self.activation_out[k], True) * error

        #print('-' * 200)
        #print('출력층 delta 값')
        #print(output_deltas)
        # 은닉 노드의 오차 함수
        hidden_deltas = []
        for HLNodeCount in self.num_yh:
            hidden_deltas.append([0.0] * HLNodeCount)
            # hidden_deltas = [0.0] * self.num_yh

        #print('-' * 200)
        #print('은닉 노드의 구조(Delta Reverse)')
        hidden_deltas.reverse()
        #print(hidden_deltas)
        temp = output_deltas  # Output Layer delta
        temp2 = copy.deepcopy(self.activation_hidden)
        temp2.reverse()
        temp3 = copy.deepcopy(self.weight_in)
        temp3.append(self.weight_out)
        temp3.reverse()  # 출력층에서 부터 입력층까지의 가중치

        #print('-' * 200)
        #print('출력층에서 부터 입력층까지의 가중치')
        #print(temp3)

        for j in range(len(hidden_deltas)):
            error = 0.0
            for k in range(len(hidden_deltas[j])):
                for i in range(len(temp)):
                    error += temp[i] * temp3[j][k][i]
                # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
                hidden_deltas[j][k] = tanh(temp2[j][k], True) * error
            temp = hidden_deltas[j]

        #print('-' * 200)
        #print('은닉층 델타 값 (Reverse)')
        #print(hidden_deltas)

        # 출력 가중치 업데이트
        for j in range(self.num_yh[len(self.num_yh) - 1]):
            for k in range(self.num_yo):
                gradient = output_deltas[k] * self.activation_hidden[len(self.activation_hidden)-1][j]
                v = mo * self.gradient_out[j][k] - lr * gradient
                self.weight_out[j][k] += v
                self.gradient_out[j][k] = gradient


        #print('-' * 200)
        #print('gradient_out')
        #print(self.gradient_out)
        temp3 = temp3[1:]
        temp4 = copy.deepcopy([self.activation_input])
        temp4.extend(self.activation_hidden)
        temp4.reverse()
        temp4 = temp4[1:]
        self.weight_in.reverse()
        #print(self.weight_in)
        self.gradient_in.reverse()
        # 출력층 제외 가중치 업데이트
        for i in range(len(temp3)):  # Weight Layer Count
            for j in range(len(hidden_deltas)):  # Except Output and Input Node
                for k in range(len(hidden_deltas[j])):
                    for l in range(len(temp4[j])):
                        gradient = hidden_deltas[j][k] * temp4[j][l]
                        v = mo * self.gradient_in[j][l][k] - lr * gradient
                        self.weight_in[j][l][k] += v
                        self.gradient_in[j][l][k] = gradient
        self.weight_in.reverse()
        self.gradient_in.reverse()

        # 오차의 계산(최소 제곱법)
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.activation_out[k]) ** 2
        return error

    # 학습 실행
    def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets)
            if i % 500 == 0:
                print('error: %-.5f' % error)

    # 결괏값 출력
    def result(self, patterns):
        for p in patterns:
            print('Input: %s, Predict: %s' % (p[0], self.update(p[0])))


if __name__ == '__main__':
    # 두 개의 입력 값, 두 개의 레이어, 하나의 출력 값을 갖도록 설정
    n = NeuralNetwork(2, [2,2,1], 1)

    # 학습 실행
    n.train(data)

    # 결괏값 출력
    n.result(data)