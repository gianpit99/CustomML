import numpy as np

from ActivationFunctions import ActivationSoftmax
from LossFunctions import LossCategoricalCrossentropy


class ActivationSoftmax_LossCategorialCrossentropy():
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()
