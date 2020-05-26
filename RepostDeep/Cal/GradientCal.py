import numpy as np

class GradientCal:
    # 활성화 함수 - 1. 시그모이드
    # 미분할 때와 아닐 때의 각각의 값
    def sigmoid(self, x, derivative=False):
        return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))  # boolean 변수는 비교 연산을 하지 않는다.

    # 활성화 함수 - 2. tanh
    # tanh 함수의 미분은 1 - (활성화 함수 출력의 제곱)
    def tanh(self, x, derivative=False):
        return 1 - x ** 2 if derivative else np.tanh(x)  # boolean 변수는 비교 연산을 하지 않는다.
