import matplotlib.pyplot as plt
import numpy as np


def prior(n=1):
    sigma_x = 1
    sigma_y = 1
    return np.random.multivariate_normal([0, 0], [[sigma_x ** 2, 0], [0, sigma_y ** 2]], n)


sigma_x = 1
sigma_y = 1
cov = [
    [sigma_x ** 2, 0],
    [0, sigma_y ** 2]
]


def prior_prob(x: list or np.ndarray) -> float:
    if type(x) == list:
        x = np.array(x)

    a = 1 / (2 * np.math.pi * sigma_x * sigma_y)
    return (a * np.exp(-0.5 * np.matmul(np.matmul(x, np.linalg.inv(cov)), x[np.newaxis].T)))[0]


def normal_pdf(v, mean, rsq):
    a = 1 / np.sqrt(2 * np.math.pi * rsq)
    p = a * np.exp(-(v - mean) ** 2 / (2 * rsq))
    return p


if __name__ == '__main__':
    k = 10  # number of reference points
    n = 1  # number of observations
    sigma = 0.5  # will be squared for variance of sensor noise
    sigma_prior = 2
    mu = 0
    reference_positions = (np.random.rand(k, 2) - 0.5) * 20
    actual_position = np.random.multivariate_normal([0, 0], cov)[np.newaxis]
    noise = np.random.normal(0, sigma ** 2, k)
    d = np.sqrt((reference_positions[:, 0] - actual_position[:, 0]) ** 2 + (
            reference_positions[:, 1] - actual_position[:, 1]) ** 2)
    r = noise + d
    # print(r)  # distances
    # text plotting
    # for a, b, c in zip(reference_positions[:, 0] - 0.5, reference_positions[:, 1] - 0.5,
    #                    [str(x + 1) for x in range(len(reference_positions))]):
    #     plt.text(a, b, c, color='blue')
    # plot reference positions as x
    # plt.plot(reference_positions[:, 0], reference_positions[:, 1], 'x')
    # plot position of car as circle (o)
    plt.plot(actual_position[:, 0], actual_position[:, 1], 'o')
    # plot lines
    # for l, length in zip(reference_positions, r):
    #     plt.plot([actual_position[0, 0], l[0]], [actual_position[0, 1], l[1]], 'k--')
    #     plt.text((actual_position[0, 0]+ l[0]) / 2, (actual_position[0, 1] + l[1]) / 2, '%.3g' % length, color='gray')
    x = np.arange(-5, 5, .25)
    y = np.arange(-5, 5, .25)
    X, Y = np.meshgrid(x, y)
    vals = np.zeros((x.shape[0], y.shape[0]))
    for i in range(len(x)):
        for j in range(len(y)):
            coords = [X[i, j], Y[i, j]]
            # new observation
            # generate new noise for every sensor
            noise = np.random.normal(0, sigma ** 2, k)
            distances = []
            for s in reference_positions:
                distances.append(np.sqrt((s[0] - coords[0]) ** 2 + (s[1] - coords[1]) ** 2))
            print(r - distances)
            val = prior_prob(coords) * np.product([normal_pdf(v, d, sigma) for v, d in zip(r, distances)])
            vals[i, j] = val
    print(vals)
    plt.contour(X, Y, vals)
    plt.axis('equal')
    plt.show()
