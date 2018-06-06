import matplotlib.pyplot as plt
import numpy as np


def prior(n=1):
    sigma_x = 1
    sigma_y = 1
    return np.random.multivariate_normal([0, 0], [[sigma_x ** 2, 0], [0, sigma_y ** 2]], n)


def prior_prob(x: list or np.ndarray) -> float:
    if type(x) == list:
        x = np.array(x)
    sigma_x = 1
    sigma_y = 1
    cov = [
        [sigma_x ** 2, 0],
        [0, sigma_y ** 2]
    ]
    a = 1 / (2 * np.math.pi * sigma_x * sigma_y)
    return (a * np.exp(-0.5 * np.matmul(np.matmul(x, np.linalg.inv(cov)), x[np.newaxis].T)))[0]


if __name__ == '__main__':
    k = 5  # number of reference points
    n = 1  # number of observations
    sigma = 1  # will be squared for variance of sensor noise
    mu = 0
    reference_positions = (np.random.rand(k, 2) - 0.5) * 20
    positions = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]])
    noise = np.random.normal(0, sigma ** 2, 5)
    r = noise + np.sqrt(
        (reference_positions[:, 0] - positions[:, 0]) ** 2 + (reference_positions[:, 1] - positions[:, 1]) ** 2)
    print(r) # distances
    # text plotting
    for a, b, c in zip(reference_positions[:, 0] - 0.5, reference_positions[:, 1] - 0.5,
                       [str(x + 1) for x in range(len(reference_positions))]):
        plt.text(a, b, c)
    # plot reference positions as x
    plt.plot(reference_positions[:, 0], reference_positions[:, 1], 'x')
    # plot position of car as circle (o)
    plt.plot(positions[:, 0], positions[:, 1], 'o')
    plt.show()
