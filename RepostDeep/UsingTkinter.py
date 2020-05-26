from tkinter import *

class UsingTkinter:
    def __init__(self, title, width, height, inputCount, hiddenCount, outputCount):
        self.root = Tk()
        self.root.title(title)
        self.root.resizable(True, True)
        self.preLayer = []  # 이전층 리스트
        self.canvas = Canvas(self.root, width=width, height=height, bg='white', bd=2)
        self.canvas.pack(fill="both", expand=True)
        self.makeText(width/2, 50, 'Neural Network Layout', color="black", font=30)
        self.makeText(width / 2, 100, 'Input Layer : ' + str(inputCount) + ', Hidden Layer : ' + str(hiddenCount)
                      + ', Output Layer : ' + str(outputCount), color="black", font=20)
        self.inputPositionCalculator(inputCount, len(hiddenCount) + 2, width)
        self.hiddenPositionCalculator(hiddenCount)
        self.outputPositionCalculator(outputCount)
        self.connectNode()  # 노드간 선 연결 함수

        self.root.mainloop()

    def inputPositionCalculator(self, inputCount, layerCount, width):
        x = width//2 - ((layerCount//2) * 200) + (70 if layerCount % 2 == 0 else 0)
        self.x, y = (x, 400)
        temp = -(50 * inputCount)
        temp2 = 50
        temp3 = 50
        y1 = y + temp
        y2 = y + temp
        layer = []
        for node in range(inputCount):
            y2 += temp2
            self.makeCircle(self.x, y1, self.x+50, y2, "blue")
            self.makeText(self.x + 25, y1 + 25, 'x' + str(node))
            layer.append([self.x, y1, self.x + 50, y2])
            y1 += temp2 + temp3
            y2 += temp3
        self.preLayer.append(layer)

    def hiddenPositionCalculator(self, hiddenCount):
        if isinstance(hiddenCount, list):
            for nodeCount in range(len(hiddenCount)):
                self.x += 200
                y = 400
                temp = -(50 * hiddenCount[nodeCount])
                temp2 = 50
                temp3 = 50
                y1 = y + temp
                y2 = y + temp
                layer =[]
                for node in range(hiddenCount[nodeCount]):
                    y2 += temp2
                    self.makeCircle(self.x, y1, self.x + 50, y2, "red")
                    self.makeText(self.x + 25, y1 + 25, 'h' + str(nodeCount + 1) + str(node))
                    layer.append([self.x, y1, self.x + 50, y2])
                    y1 += temp2 + temp3
                    y2 += temp3
                self.preLayer.append(layer)
        else:
            self.x += 200
            y = 400
            temp = -(50 * hiddenCount)
            temp2 = 50
            temp3 = 50
            y1 = y + temp
            y2 = y + temp
            layer = []
            for node in range(hiddenCount):
                y2 += temp2
                self.makeCircle(self.x, y1, self.x + 50, y2, "red")
                self.makeText(self.x + 25, y1 + 25, 'h' + str(node))
                layer.append([self.x, y1, self.x + 50, y2])
                y1 += temp2 + temp3
                y2 += temp3
            self.preLayer.append(layer)

    def outputPositionCalculator(self, outputCount):
        self.x += 200
        y = 400
        temp = -(50 * outputCount)
        temp2 = 50
        temp3 = 50
        y1 = y + temp
        y2 = y + temp
        layer = []
        for node in range(outputCount):
            y2 += temp2
            self.makeCircle(self.x, y1, self.x + 50, y2, "green")
            self.makeText(self.x + 25, y1 + 25, 'y' + str(node))
            layer.append([self.x, y1, self.x + 50, y2])
            y1 += temp2 + temp3
            y2 += temp3
        self.preLayer.append(layer)

    def connectNode(self):
        for layerCount in range(len(self.preLayer)-1):
            currentLayer = self.preLayer[layerCount]
            nextLayer = self.preLayer[layerCount + 1]
            for currentLayerNode in range(len(currentLayer)):
                x1 = currentLayer[currentLayerNode][2]
                y1 = currentLayer[currentLayerNode][3] - 25
                for nextLayerNode in range(len(nextLayer)):
                    x2 = nextLayer[nextLayerNode][0]
                    y2 = nextLayer[nextLayerNode][1] + 25
                    self.makeLine(x1, y1, x2, y2)

    def makeCircle(self, x1, y1, x2, y2, color):
        self.canvas.create_oval(x1, y1, x2, y2, fill=color)

    def makeLine(self, x1, y1, x2, y2):
        self.canvas.create_line(x1, y1, x2, y2, fill="black")

    def makeText(self, x1, y1, content, color='white' , font=12):
        self.canvas.create_text(x1, y1, text=content, font=("나눔고딕코딩", font), fill=color)
