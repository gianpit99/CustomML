import numpy as np

# Base Loss Class
class Loss:
    def regularizationLoss(self):
        regularizationLoss = 0

        for layer in self.trainableLayers:
            if layer.weightRegularizerL1 > 0:
                regularizationLoss += layer.weightRegularizerL1 * np.sum(np.abs(layer.weights))

            if layer.weightRegularizerL2 > 0:
                regularizationLoss += layer.weightRegularizerL2 * np.sum(layer.weights * layer.weights)

            if layer.biasRegularizerL1 > 0:
                regularizationLoss += layer.biasRegularizerL1 * np.sum(layer.biases * layer.biases)

            if layer.biadRegularizerL2 > 0:
                regularizationLoss += layer.biasRegularizerL2 * np.sum(layer.biases * layer.biases)

        return regularizationLoss

    def rememberTrainableLayers(self, trainableLayers):
        self.trainableLayers = trainableLayers

    def calculate(self, output, y, *, includeRegularization=False):
        # Calculate the loss for each sample
        sampleLoss = self.forward(output, y)

        # Calculate the mean loss
        meanLoss = np.mean(sampleLoss)

        self.accumulatedSum += np.sum(sampleLoss)
        self.accumulatedCount += len(sampleLoss)

        if not includeRegularization:
            return meanLoss

        # Return the loss
        return meanLoss, self.regularizationLoss()

    def calculateAccumulated(self, *, includeRegularization=False):
        dataLoss = self.accumulatedSum / self.accumulatedCount

        if not includeRegularization:
            return dataLoss

        return dataLoss, self.regularizationLoss()

    def newPass(self):
        self.accumulatedSum = 0
        self.accumulatedCount = 0


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



class LossBinaryCrossentropy(Loss):
    def forward(self, yPred, yTrue):
        yPredClipped = np.clip(yPred, 1e-7, 1 - 1e-7)

        sampleLoss = -(yTrue * np.log(yPredClipped) + (1 - yTrue) * np.log(1 - yPredClipped))

        sampleLoss = np.mean(sampleLoss, axis=1)

        return sampleLoss

    def backward(self, dValues, yTrue):
        samples = len(dValues)

        outputs = len(dValues[0])

        clippedDValues = np.clip(dValues, 1e-7, 1 - 1e-7)

        self.dInputs = -(yTrue / clippedDValues - (1 - yTrue) / (1 - clippedDValues)) / ouput

        self.dInputs / samples
