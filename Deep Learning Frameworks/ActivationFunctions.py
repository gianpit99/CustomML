import numpy as np


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
