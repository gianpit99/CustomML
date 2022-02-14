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
