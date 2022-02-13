import numpy as np

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
