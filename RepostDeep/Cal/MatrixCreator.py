import numpy as np

class MatrixCreator:

    # 활성화 값 배열 만드는 함수
    def makeActivation(self, i, h, o):
        '''
        :param i: Input Layer Node
        :param h: Hidden Layer List
        :param o: Output Layer Node
        :return:
        '''
        d = {}
        # 입력층
        d[0] = np.ones(i)
        # 은닉층
        for layerCount, nodeCount in enumerate(h):
            d[layerCount + 1] = np.ones(nodeCount)
        # 출력층
        d[len(h) + 1] = np.ones(o)

        return d

    # 가중치 배열 만드는 함수
    def makeWeight(self, i, h, o):
        '''
        :param i: Input Layer Node
        :param h: Hidden Layer List
        :param o: Output Layer Node
        :param fill: default 0.0
        :return:
        '''
        d = {}
        # 입력층과 은닉층 사이 가중치
        for inp in range(i):
            d[0] = np.random.rand(i*h[0]).reshape(i, h[0])
        # 이외의 가중치
        maxN = len(h)
        for layerCount, nodeCount in enumerate(h):
            if maxN - 1 > layerCount:
                d[layerCount + 1] = np.random.rand(nodeCount*h[layerCount + 1]).reshape(nodeCount, h[layerCount + 1])
            else:
                d[layerCount + 1] = np.random.rand(nodeCount*o).reshape(nodeCount, o)

        return d

    def makeBias(self, h, o):
        '''
        :param h: Hidden Layer List
        :param o: Output Layer Node
        :return:
        '''
        d = {}
        for layerCount, nodeCount in enumerate(h):
            d[layerCount] = np.ones(nodeCount)
        d[len(h)] = np.ones(o)

        return d

    def makeDeltaMetrix(self, h, o):
        '''
        :param h: Hidden Layer List
        :param o: Output Layer Node
        :return:
        '''

        d = {}
        d[len(h)+1] = np.zeros(o)

        if type(h) == list:
            for layerCount, nodeCount in enumerate(h):
                d[layerCount + 1] = np.zeros(nodeCount)
        else:
            d[1] = np.zeros(h)

        return d