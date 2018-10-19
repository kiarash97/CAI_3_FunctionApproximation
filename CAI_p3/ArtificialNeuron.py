import numpy as np
import math

class Artificial_Neuron:
    def __init__(self, inputs_num ,activation_function= "sigmo" ):
        self.inputWeights = np.zeros(inputs_num)
        self.biasWeight = 0
        # self.inputWeights = np.zeros(inputs_num+1)
        self.inputNumber = inputs_num


        self.activationFunction = activation_function
        self.input = None
    def output(self):
        if self.activationFunction == "sigmo":
            return self.sigmoid(self.sumFunction())

    def sumFunction(self):
        # return np.dot(self.inputWeights , self.input)
        return np.dot(self.inputWeights, self.input) -self.biasWeight

    def setInput(self, input):
        # self.input = np.append(input , -1)
        self.input = input

    def randomWeightsInitialize(self):
        self.inputWeights = np.random.rand(self.inputNumber)
        self.biasWeight = np.random.rand()

    def setWeights(self, weights , bias_weight=None):
        self.inputWeights = weights
        if not bias_weight :
            self.biasWeight = bias_weight

    def show(self):
        print ("input weights = " , self.inputWeights,"bias=",self.biasWeight)
        # print("input weights = ", self.inputWeights)
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

# i = np.array([1,2,3])
# x = Artificial_Neuron(inputs_num=3)
# x.randomWeightsInitialize()
# x.setInput(i)
# print(x.inputWeights, x.biasWeight)
# print (x.sumFunction())
# print (x.output())




