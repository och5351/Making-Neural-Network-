<h1 align="center">
   Making deep neural networks in Python
</h1>
<p align="center">No tensorflow</p>
Enviorment
===
    - Using Python 3.6
    - Using Numpy 
    - Using Keras(MNIST)

Detail description
===

파이썬으로 구현한 Deep Neural Network 입니다. 아직 Loss가 많이 튀는 그래프를 나타내고 있어 어느 함수에 문제가 있는지 확인해봐야겠지만 기본적인 오차역전파법의 공식(출력층에서 가중치 업데이트, 은닉층에서의 가중치 업데이트)은 정확히 구현하였습니다.

Design
---
<p align="center">
  <img src="https://user-images.githubusercontent.com/45858414/78989755-7a337380-7b6f-11ea-8ff3-a001ffee727c.PNG" width="80%">
</p>

Chain Rull
---
    left = Hidden - Output         right = Input - Hidden

### ex)
<p align="center">
   <img src="https://user-images.githubusercontent.com/45858414/78990864-7ead5b80-7b72-11ea-8915-983fd07ac353.PNG" width="50%">
</p>
<p algin="center">
   <img src="https://user-images.githubusercontent.com/45858414/78990611-d13a4800-7b71-11ea-8b89-23837034fa11.PNG" width="49%">
   <img src="https://user-images.githubusercontent.com/45858414/78990678-08a8f480-7b72-11ea-8882-40aa49d9cf0c.PNG" width="49%">
</p>

Loss
---
<p align="center">
   <img src="https://user-images.githubusercontent.com/45858414/78989919-e8783600-7b6f-11ea-9015-880a290a44ea.png" width="33%">
   <img src="https://user-images.githubusercontent.com/45858414/78989944-ffb72380-7b6f-11ea-99c6-84515fbc6458.png" width="33%">
</p>
