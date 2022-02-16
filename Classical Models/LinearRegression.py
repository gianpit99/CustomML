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
        return np.matmul(self.b, x)

    def error(self, x, y):
        return np.power((self.predict(x) - y), 2)



def getClustedData2D(nSamples=400):
    center1 = (50, 60)
    center2 = (80, 20)
    distance = 20


    x1 = np.random.uniform(center1[0], center1[0] + distance, size=(nSamples,1))
    y1 = np.random.normal(center1[1], distance, size=(nSamples,1))

    x2 = np.random.uniform(center2[0], center2[0] + distance, size=(nSamples,1))
    y2 = np.random.normal(center2[1], distance, size=(nSamples,1))

    res_a = np.append(x1, y1, axis=1)
    res_a = np.append(res_a, np.zeros((nSamples,1)), axis=1)

    res_b = np.append(x2, y2, axis=1)
    res_b = np.append(res_b, np.ones((nSamples,1)), axis=1)

    res = np.append(res_a, res_b, axis=0)

    return res



def main():
    ClusterData2D = getClustedData2D()

    plt.scatter(ClusterData2D[ClusterData2D[:, 2] == 1][:, 0], ClusterData2D[ClusterData2D[:, 2] == 1][:, 1])
    plt.scatter(ClusterData2D[ClusterData2D[:, 2] == 0][:, 0], ClusterData2D[ClusterData2D[:, 2] == 0][:, 1])
    plt.show()


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
