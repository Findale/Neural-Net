import csv
import math
from random import seed
from random import random
from datetime import datetime

class Layer:

    def __init__(self, prevLayer = None, nextLayer = None):
        self.neurons = []
        self.yPreds = []
        self.delts = []
        self.length = 0
        self.prev = prevLayer
        self.next = nextLayer

    def addNeuron(self, newNeuron):
        self.neurons.append(newNeuron)
        self.length += 1
        self.yPreds.append(0)
        self.delts.append(0)

    def addPrev(self, prevLayer):
        self.prev = prevLayer

    def addNext(self, nextLayer):
        self.next = nextLayer

    def getPreds(self):
        return self.yPreds

    def getLength(self):
        return self.length

    def dataDistr(self, xVals):
        for i in range(self.length):
            self.yPreds[i] = self.neurons[i].hiddenFeed(xVals)

    def outDistr(self, xVals, yVals):
        self.dataDistr(xVals)
        for i in range(self.length):
            self.delts[i] = self.neurons[i].outDelt(self.yPreds[i], yVals[i])
        self.backProp()

    def backProp(self):
        weights = []
        for node in self.neurons:
            weights.append(node.getWeight())
        self.prev.hidBack(weights, self.delts)
    
    def hidBack(self, weights, deltas):
        for i in range(self.length):
            weightArgs = []
            for weight in weights:
                weightArgs.append(weight[i])
            self.neurons[i].deltCalc(self.yPreds[i], deltas, weightArgs)
        self.next.update()
    
    def update(self):
        for i in range(self.length):
            self.neurons[i].updateWeights(self.yPreds[i], self.delts[i])


class Neuron:
    alpha = 0.1

    def __init__(self, count):
        self.theta = 0.1
        self.w = [random() for i in range(count)]

    def hiddenFeed(self, data):
        bigX = 0.0
        for index in range(len(self.w)):
            bigX += (float(data[index]) * self.w[index])
        bigX -= self.theta
        yPred = 1 / (1 + math.exp(-(bigX)))
        return yPred

    def outDelt(self, yPred, yVal):
        yVal = float(yVal)
        delta = yPred * (1 - yPred) * (yVal - yPred)
        return delta
        
    def deltCalc(self, yPred, deltas, weights):
        summer = 0
        for i in range(len(deltas)):
            summer += deltas[i]*weights[i]
        delt = yPred * (1 - yPred) * summer
        self.updateWeights(yPred, delt)

    def getWeight(self):
        return self.w

    def updateWeights(self, yPred, delta):
        for weight in self.w:
            weight += Neuron.alpha * yPred * delta
        self.theta -= Neuron.alpha * delta



def main(file, dataLen, categories):
    seed(datetime.now())

    data = []
    with open(file, "r") as csvfile:
        for row in csv.reader(csvfile, delimiter = ','):
            data.append(row)
    hidden = Layer()
    output = Layer()
    hidden.addNext(output)
    output.addPrev(hidden)
    for i in range(dataLen):
        hidden.addNeuron(Neuron(dataLen))
    for i in range(categories):
        output.addNeuron(Neuron(dataLen))
    
    totalNum = float(len(data))
    MAD = 0
    for epoch in range(20):
        i = 0
        totalSum = 0
        MAD = 0
        for sample in data:
            i += 1
            xVals = sample[:dataLen]
            yVals = sample[dataLen:]
            hidden.dataDistr(xVals)
            hiddenYs = hidden.getPreds()
            output.outDistr(hiddenYs, yVals)
            outYs = output.getPreds()
            #print("Epoch " + str(epoch) + ", Iteration " + str(i) + ": Prediction is ")
            #print('[%s]' % ', '.join(map(str, outYs)))
            for j in range(len(yVals)):
                totalSum += abs(float(yVals[j]) - outYs[j])
        MAD = totalSum / len(data)
        print("Epoch " + str(epoch+1) + " Results: MAD = " + str(MAD))

print("IRIS DATASET")
print("____________")
main('./datasets/iris.csv', 4, 3)

print("\n\nWINE DATASET")
print("____________")
main('./datasets/wine.csv', 13, 3)

print("\n\nBREAST CANCER DATASET")
print("_____________________")
main('./datasets/breastcancer.csv', 10, 2)