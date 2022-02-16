import numpy as np
import matplotlib.pyplot as plt



class SVM:
    '''
    Terminology
    -----------

    Support Vector : Training data that is closest to hyperplane
    Margin : distance from support vector to hyperplane


    Hyper Plane is a D-1 Seperator
    H: W^T * X + b = 0

    W is the normal of the hyperplanen
    W^T * X is the projection of a point on the hyperplane

    W^T * X + b > 0 for group 1
    W^T * X + b < 0 for group 2


    Objective
    ---------
    - SVM's maximize the distance of the hyperplane to the closest points (or the minimum distance)

    Wolfe Dual Problem
    ------------------


    '''
    def __init__(self, C=1000, gamma=0, kernel='None'):
        self.supportVectors = None
        self.w = None
        self.b = None

        self.C = C
        self.gamma = gamma
        self.kernal = None

        self.n = None

    def fit(self, x, y, lr=1e-3, sc=1e-9):
        if self.kernal == 'rbf':
            pass
        elif self.kernal == 'linear':
            pass

        self.n, d = x.shape
        self.w = np.random.randn(d)
        self.b = 0

        prevLoss = np.inf
        loss = 0
        while (abs(prevLoss - loss) >= sc):
            margin = self.margin(x, y)
            prevLoss = loss
            loss = self.loss(margin)
            #print(loss)


            idxMissClassified = np.where(margin< 1)[0]
            dw = self.w - self.C * np.dot(y[idxMissClassified], x[idxMissClassified])
            self.w = self.w - lr * dw

            db = - self.C * np.sum(y[idxMissClassified])
            self.b = self.b - lr * db

        self.supportVectors = np.where(self.margin(x, y) <= 1)[0]

    def predict(self, x):
        return np.sign(self.decision(x))

    def decision(self, x):
        return np.dot(x, self.w) + self.b

    def loss(self, margin):
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(np.maximum(0, 1 - margin))

    def margin(self, x, y):
        return y * self.decision(x)

    def KernalRBF(self, sigma=np.inf):
        return

    def KernalLinear(self, n=1):
        return

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
    x = ClusterData2D[:, 0:2]
    y = ClusterData2D[:, 2]

    y[y==0] = -1

    svmModel = SVM()
    svmModel.fit(x, y)

    yPred = svmModel.predict(x)

    for y, yp in zip(y, yPred):
        print(y, yp)

    plt.scatter(x[ClusterData2D[:, 2]==1][:, 0], x[ClusterData2D[:, 2]==1][:, 1])
    plt.scatter(x[ClusterData2D[:, 2]==-1][:, 0], x[ClusterData2D[:, 2]==-1][:, 1])
    plt.show()

    plt.scatter(x[yPred==1][:, 0], x[yPred==1][:, 1])
    plt.scatter(x[yPred==-1][:, 0], x[yPred==-1][:, 1])
    plt.show()

if __name__ == "__main__":
    main()
