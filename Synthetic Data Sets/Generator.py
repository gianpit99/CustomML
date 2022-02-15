import numpy as np
import matplotlib.pyplot as plt


def getSpiralData2D(nSamples = 400):
    theta = np.sqrt(np.random.rand(nSamples))*2*np.pi # np.linspace(0,2*pi,100)

    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(nSamples,2)

    r_b = -2*theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(nSamples,2)

    res_a = np.append(x_a, np.zeros((nSamples,1)), axis=1)
    res_b = np.append(x_b, np.ones((nSamples,1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    return res

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
    SprialData2D = getSpiralData2D()

    plt.scatter(SprialData2D[SprialData2D[:, 2] == 1][:, 0], SprialData2D[SprialData2D[:, 2] == 1][:, 1])
    plt.scatter(SprialData2D[SprialData2D[:, 2] == 0][:, 0], SprialData2D[SprialData2D[:, 2] == 0][:, 1])
    plt.show()


    ClusterData2D = getClustedData2D()

    plt.scatter(ClusterData2D[ClusterData2D[:, 2] == 1][:, 0], ClusterData2D[ClusterData2D[:, 2] == 1][:, 1])
    plt.scatter(ClusterData2D[ClusterData2D[:, 2] == 0][:, 0], ClusterData2D[ClusterData2D[:, 2] == 0][:, 1])
    plt.show()



if __name__ == "__main__":
    main()
