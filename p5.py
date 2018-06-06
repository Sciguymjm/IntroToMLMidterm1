import numpy as np
import matplotlib


if __name__ == '__main__':
    k = 5 # number of reference points
    n = 1 # number of observations
    sigma = 1
    mu = 0
    reference_positions = []
    for x in range(k):
        x_val = np.random.randint(-10, 10)
        y_val = np.random.randint(-10, 10)
        reference_positions.append([x_val, y_val])

    for x in range(n):
        x_val = np.random.randint(-10, 10)
        y_val = np.random.randint(-10, 10)
        r = np.sqrt((reference_positions[:, 0] - x_val)**2 + (reference_positions[:, 1] - y_val)**2)
        print (r)
    np.random.multivariate_normal([[0], [0]], [[1,0],[0,1]])