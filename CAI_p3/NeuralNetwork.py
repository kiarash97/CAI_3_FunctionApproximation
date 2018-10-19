import numpy as np
from ArtificialNeuron import Artificial_Neuron

class FeedForwardNN():
    def __init__(self , input_size , cost_function = "SSE" , learning_rate = 1 , epochs = 1000):
        self.layers = []
        self.inputSize = input_size
        self.layersCount = 0
        self.learningRate = learning_rate
        self.epochs = epochs
        self.min = 0
    def addLayer(self , layer_size , random_initialize_flag =True):
        temp = []
        if self.layersCount == 0:
            for _i in range(layer_size):
                x = Artificial_Neuron(self.inputSize)
                if random_initialize_flag :
                    x.randomWeightsInitialize()
                temp.append(x)
        else:
            for _i in range(layer_size):
                x = Artificial_Neuron(len(self.layers[self.layersCount-1]))
                if random_initialize_flag :
                    x.randomWeightsInitialize()
                temp.append(x)

        self.layers.append(temp)
        self.layersCount+=1

    def showLayer(self, layer_number):
        for i in self.layers[layer_number]:
            i.show()

    def showNetwork(self):
        print ("input layer has",self.inputSize,"inputs\n")
        for counter in range(len(self.layers)):
            print("layer ",counter ,"input weights")
            self.showLayer(counter)
            print ("\n")
            counter+=1

    def getLayer(self, layer_number):
        return self.layers[layer_number]

    def setInput(self , input):
        self.input = input
        for i in self.layers[0]:
            i.setInput(self.input)

    def networkOutput(self):
        layer_counter = 0
        #connect layers
        for layer in self.layers:
            if layer_counter : #if not layer == 0
                for neuron in layer:
                    prevLayer = self.getLayer(layer_counter-1)
                    temp = []
                    for prevneuron in prevLayer:
                        temp.append(prevneuron.output())
                    neuron.setInput(np.asarray(temp))

            layer_counter+=1

        #calculate output
        outputLayer = self.getLayer(self.layersCount-1)
        outputList = []
        for neuron in outputLayer:
            outputList.append(neuron.output())
        return outputList


    def train(self , patterns , true):
        for _i in range(self.epochs):
            counter = 0
            self.eval(patterns , true)
            for p in patterns :
                self.true = true[counter]
                counter +=1
                self.setInput(p)
                for layer in range(self.layersCount) :
                    neuron_number = 0
                    delta_bias = self.learningRate * self.deltaBias(layer-1)
                    for neuron in self.getLayer(layer):
                        delta_weight = -1 * self.learningRate * neuron.output() * self.deltaFunction(layer, neuron_number)
                        neuron_number+=1
                        self.updateWeight( neuron , delta_weight , delta_bias)


    def updateWeight(self , neuron , delta_weight , delta_bias):
        neuron.setWeights( weights= neuron.inputWeights + delta_weight , bias_weight= neuron.biasWeight +delta_bias)

    def eval(self, patterns , true):
        x=0
        for i in range(len(patterns)) :
            self.setInput(patterns[i])
            output= self.networkOutput()
            t = true[i]
            x+= output - t
        print(x)
    def costFunction(self  ):
        output = self.networkOutput()
        error = 0
        for i in range(len(output)):
            error+= (self.true[i]-output[i])**2
        error *= (1/2)*(1/len(output))
        return error

    def deltaFunction(self , layerNumber , neuronNumber ):
        output = self.networkOutput()
        layer = self.getLayer(layerNumber)
        if layerNumber == self.layersCount -1 : #output layer
            # ***remember to complete
            delta_neuronNumber = -(self.true[neuronNumber] - output[neuronNumber]) \
                                 * self.actDerivation(layer[neuronNumber])
            # ***until here

        else: #hidden layer
            nextLayer = self.getLayer(layerNumber+1)
            delta_neuronNumber = 0
            for neuNumber in range(len(nextLayer)) :
                delta_neuronNumber += self.deltaFunction(layerNumber+1, neuNumber )\
                                      * nextLayer[neuNumber].inputWeights[neuronNumber] \
                                      *self.actDerivation(layer[neuronNumber])

        return delta_neuronNumber

    def deltaBias(self , layerNumber):
        if layerNumber != self.layersCount -1 :
            nextLayer = self.getLayer(layerNumber+1)
            delta_bias = 0
            for neuNumber in range(len(nextLayer)):
                delta_bias += self.deltaFunction(layerNumber+1 , neuNumber) * nextLayer[neuNumber].biasWeight
        else :
            return 0
        return delta_bias



    def actDerivation(self , neuron):
        if neuron.activationFunction =="sigmo":
            return (1-neuron.output()) * neuron.output()

    def predict(self, input):
        self.setInput(input)
        return self.networkOutput()
    def test(self):
        for i in self.layers[0]:
            print (i.input)
            print (i.inputWeights)

        print ("Hi")
        for i in self.layers[1]:
            print (i.input)
            print (i.inputWeights)



# x = FeedForwardNN(input_size=2)
# patterns = np.array([[0,0],[0,1],[1,0],[1,1]])
# true = np.array([[1],[0],[0],[1]])
# x.addLayer(2)
# x.addLayer(1)
# # patterns = np.array([[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3]])
# # true = np.array([[0],[0],[0],[0],[1],[1],[1],[1]])
# # x.addLayer(3)
# # x.addLayer(1)
# x.train(patterns,true)
#
# print (x.predict(np.array([0,0])))
# print (x.predict(np.array([1,1])))
# print (x.predict(np.array([0,1])))
# print (x.predict(np.array([1,0])))

# x.test()

import sklearn.datasets
import matplotlib.pyplot as plt

x =FeedForwardNN(input_size=2)

np.random.seed(0)
X, y = sklearn.datasets.make_moons(n_samples=100, noise=0.20)
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# print (X,np.array([y]))
x.addLayer(layer_size=5)
x.addLayer(layer_size=2)
i = np.array([[1,1],[0,1],[1,0],[0,0]])
t = np.array([[1,1],[0,0],[0,0],[1,1]])
# x.showNetwork()
x.train(i , t)
x.predict(np.array([1,1]))
# x.showNetwork()