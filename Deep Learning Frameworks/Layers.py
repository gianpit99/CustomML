import numpy as np


class LayerDense:
    def __init__(   self,
                    nInputs,
                    nNeurons,
                    weightRegularizationL1=0.0,
                    weightRegularizationL2=0.0,
                    biasRegularizationL1=0.0,
                    biasRegularizationL2=0.0):
        self.weights = 0.01 * np.random.randn((nInputs, nNeurons), dtype=np.float32)
        self.biases = np.zeros((1, nNeurons))

        self.weightRegularizationL1 = weightRegularizationL1
        self.weightRegularizationL2 = weightRegularizationL2
        self.biasRegularizationL1 = biasRegularizationL1
        self.biasRegularizationL2 = biasRegularizationL2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dValues):
        # Weight and bias gradients
        self.dWeights = np.dot(self.inputs.T, dValues)
        self.dBiases = np.sum(dValues, axis=0, keepdims=True)

        if self.weightRegularizationL1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dWeights += self.weightRegularizationL1 * dL1

        if self.weightRegularizationL2 > 0:
            self.dWeights += 2 * self.weightRegularizationL2 * self.weights

        if self.biasRegularizationL1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dBiases += self.biasRegularizationL1 * dL1

        if self.biasRegularizationL2 > 0:
            self.dBiases += 2 * self.biasRegularizationL2 * self.biases

        # Value gradients for chain rule
        self.dInputs = np.dot(dValues, self.weights.T)

    def getParameters(self):
        return self.weights, self.biases

    def setParameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class LayerDropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binaryMask = np.random.binomial(1, self.rate, size=inputs.shape / self.rate)

    def backward(self, dvalues):
        seld.dinputs = dvalues * self.binaryMask


class LayerInput:
    def forward(self, Inputs, Training):
        self.output = inputs
