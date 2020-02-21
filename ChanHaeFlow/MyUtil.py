import numpy as np

class MyUtil:

    def to_Catgorical(self, x, i = 0):
        result = np.arange(len(x) * 10).reshape(len(x), 10)
        for k in range(len(x)):
            if i == 0:
                i = x[k]

            for j in range(i):
                if j == x[k]-1:
                    result[k][j] = 1
                else:
                    result[k][j] = 0
        return result

    def pprint(self, arr):
        print("type:{}".format(type(arr)))
        print("shape: {}, dimension: {}, dtype:{}".format(arr.shape, arr.ndim, arr.dtype))
        print("Array's Data:\n", arr)

