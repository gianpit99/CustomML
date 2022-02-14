import numpy as np

from ActivationFunctions import ActivationSoftmax
from LossFunctions import LossCategoricalCrossentropy


class ActivationSoftmax_LossCategorialCrossentropy():

    def backward(self, dValues, yTrue):
        nSamples = len(dValues)

        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis=1)

        self.dInputs = dValues.copy()

        self.dInputs[range(samples), yTrue] -= 1
        self.dInputs = self.dInputs / samples
