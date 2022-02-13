import numpy as np


class OptimizerSGD:
    def __init__(self, learningRate=1.0, decay=0.0):
        self.learningRate = learningRate
        self.decay = decay
        self.currentLearningRate = learningRate
        self.iterations = 0
