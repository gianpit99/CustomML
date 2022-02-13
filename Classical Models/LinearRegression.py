import numpy as np


class LinearRegression:
    def __init__(self, fitIntercept=True):
        self.fitIntercept = fitIntercept
        self.error = None
        self.b = None

    def fit(self, x, y):
        self.b = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
        return self.b

    def predict(self, x):
        return

    def error(self, x, y):
        return




def main():
    X = np.linspace(1, 100, 100)
    Y = 0.8 * X + 10 * np.random.random((1, 100)) - 10 * np.random.random((1, 100))

    import matplotlib.pyplot as plt

    print(X.shape, Y.reshape(-1, 1).shape)

    plt.plot(X.reshape(-1, 1), Y.reshape(-1, 1))
    plt.show()


    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
    print(model.b)




    return


if __name__ == "__main__":
    main()
