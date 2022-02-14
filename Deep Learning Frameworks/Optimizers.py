import numpy as np


class OptimizerSGD:
    def __init__(   self,
                    learningRate=1.0,
                    decay=0.0,
                    momentum=0.0):
        self.learningRate = learningRate
        self.decay = decay
        self.currentLearningRate = learningRate
        self.iterations = 0
        self.momentum = momentum

    def preUpdateParams(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1.0 / (1.0 + self.decay * self.iterations))

    def updateParams(self, layer):
        if self.momentum:

            if not hasattr(layer, 'weightMomentums'):
                layer.weightMomentums = np.zeros_like(layer.weights)

                layer.biasMomentums = np.zeros_like(layer.biases)

            weightUpdates = self.momentum * layer.weightMomentums - self.currentLearningRate * layer.dWeights
            layer.weightMomentums = weightUpdates

            biasUpdates = self.momentum * layer.biasMomentums - self.currentLearningRate * layer.dBiases
            layer.biasMomentums = biasUpdates
        else:
            weightUpdates = -self.currentLearningRate * layer.dWeights
            biasUpdates = -self.currentLearningRate * layer.dBiases

        layer.weights += weightUpdates
        layer.biases += biasUpdates

    def postUpdateParams(self):
        self.iternations += 1


class OptimizerRMSprop:
    def __init__(   self,
                    learningRate=0.001,
                    decay=0.0,
                    epsilon=1e-7):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iternations = 0
        self.epsilon = epsilon
        self.rho = rho

    def preUpdateParams(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1.0 / (1.0 + self.decay * self.iterations))

    def updateParams(self, layer):
        if not hasattr(layer, 'weightCache'):
            layer.weightCache = np.zeros_like(layer.weights)
            layer.biasCahce = np.zeros_like(layer.biases)

        layer.weightCache = self.rho * layer.weightCache + (1 - self.rho) * layer.dWeights**2
        layer.biasCahce = self.rho * layer.biasCahce + (1 - self.rho) * layer.dBiases**2

        layer.weights += -self.currentLearningRate * layer.dWeights / (np.sqrt(layer.weightCache) + self.epsilon)
        layer.biases += -self.currentLearningRate * layer.dBiases / (np.sqrt(layer.biasCahce) + self.epsilon)

    def postUpdateParams(self):
        self.iterations += 1

class OptimizerAdam:
    def __init__(   self,
                    learningRate=0.001,
                    decay=0.0,
                    epsilon=1e-7,
                    beta1 = 0.9,
                    beta2 = 0.999):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iternations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def preUpdateParams(self):
        if self.decay:
            self.currentLearningRate = self.learningRate * (1.0 / (1.0 + self.decay * self.iterations))

    def updateParams(self, layer):

        if not hasattr(layer, 'weightCache'):
            layer.weightMomentums = np.zeros_like(layer.weights)
            layer.weightCache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.biasCahce = np.zeros_like(layer.biases)

        layer.weightMomentums = self.beta1 * layer.weightMomentums + (1 - self.beta1) * layer.dWeights
        layer.biasMomentums = self.beta1 * layer.biasMomentums + (1 - self.beta1) * layer.dBiases


        weightMomentumsCorrected = layer.weightMomentums / (1 - self.beta1 ** (self.iterations + 1))
        biasMomentumsCorrected = layer.biasMomentums / (1 - self.beta1 ** (self.iterations + 1))

        layer.weightCache = self.beta2 * layer.weightCache + (1 - self.beta2) * layer.dWeights**2
        layer.biasCahce = self.beta2 * layer.biasCahce + (1 - self.beta2) * layer.dBiases**2

        weightCacheCorrected = layer.weightCache / (1 - self.beta2 ** (self.iterations + 1))
        biasCahceCorrected = layer.biasCahce / (1 - self.beta2 ** (self.iterations + 1))

        layer.weights += -self.currentLearningRate * weightMomentumsCorrected / (np.sqrt(weightCacheCorrected) + self.epsilon)
        layer.biases += -self.currentLearningRate * biasMomentumsCorrected / (np.sqrt(biasCahceCorrected) + self.epsilon)


    def postUpdateParams(self):
        self.iterations += 1
