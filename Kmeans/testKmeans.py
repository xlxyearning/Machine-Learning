from Kmeans import Kmeans
import numpy as np

if __name__  == "__main__":
    x1 = np.array([1, 1])
    x2 = np.array([2, 1])
    x3 = np.array([4, 3])
    x4 = np.array([5, 4])
    testX = np.vstack((x1, x2, x3, x4))

    kMeans = Kmeans(2, 3, testX)
    kMeans.kmeans()