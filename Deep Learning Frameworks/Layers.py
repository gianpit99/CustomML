import numpy as np


class LayerDense:
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.01 * np.random.randn((nInputs, nNeurons), dtype=np.float32)
        self.biases = np.zeros((1, nNeurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dValues):
        # Weight and bias gradients
        self.dWeight = np.dot(self.inputs.T, dValues)
        self.dBiases = np.sum(dValues, axis=0, keepdims=True)

        # Value gradients for chain rule
        self.dInputs = np.dot(dValues, self.weights.T)






class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dValues):
        # Make a copy since we need to edit the original variables
        self.dInputs = dValues.copy()

        # Make the gradient zero when values are negative
        self.dInputs[self.inputs <= 0] = 0

class ActivationSoftmax:
    def forward(self, inputs):
        # Un-Normalized Probabilities
        expValues = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize Each Sample
        probs = expValues / np.sum(exp_values, axis=1, keepdims=True)

        # Create the output
        self.output = probs

    def backward(self, dValues):
        # Create an array to hold the gradients
        self.dInputs = np.empty_like(dValues)

        # Enumerate and iterate over the outputs and gradients
        for index, (singleOutput, singleDValues) in enumerate(zip(self.output, dValues))
            # Flatten the output array
            singleOutput = singleOutput.reshape(-1, 1)

            # Calculate the jacobian matrix of the output
            jacobianMatrix = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)

            # Calculate the sample wise gradient
            self.dInputs[index] = np.dot(jacobianMatrix, singleDValues)






# Base Loss Class
class Loss:
    def calculate(self, output, y):
        # Calculate the loss for each sample
        sampleLoss = self.forward(output, y)

        # Calculate the mean loss
        meanLoss = np.mean(sampleLoss)

        # Return the loss
        return meanLoss


class LossCategoricalCrossentropy(Loss):
    def forward(self, yPred, yTrue):
        # The number of samples in the batch
        nSamples = len(yPred)

        # Clip the data to prevent division by zero errors
        yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)

        # Target probabilities for categorial data
        if len(yTrue.shape) == 1:
            correctConfidences = yPredClipped[
                range(samples),
                yTrue
            ]

        # Mask values if one-hot-encoding
        elif len(yTrue.shape) == 2:
            correctConfidences = np.sum(
                yPredClipped * yTrue,
                axis = 1
            )

        # Loss for each class
        negativeLogLiklihoods = -np.log(correctConfidences)
        return negativeLogLiklihoods

    def backward(self, dValues, yTrue):
        # Number of samples
        nSamples = len(dValues)

        # Number of labels in each sample
        nLables = len(dValues[0])

        # If lables are sparse, turn them into a one-hot-vector
        if len(yTrue.shape) == 1:
            yTrue = np.eye(nLabels)[yTrue]

        # Calculate the gradient
        self.dInputs = -yTrue / dValues

        # Normalize the gradient
        self.dInputs = self.dInputs / samples
