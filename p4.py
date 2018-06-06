import matplotlib.pyplot as plt
import numpy as np


def mpl(mean=None, cov=None, color='b'):
    if mean is None:
        mean = [0, 0]
    if cov is None:
        cov = [[1, 10], [10, 10]]
    x, y = np.random.multivariate_normal(mean, cov, 5000).T
    # print('mean', np.mean(x), np.mean(y))
    # print('covariance', np.cov(x, y))
    plt.plot(x, y, 'x', color=color)


def mpl_show():
    plt.axis('equal')

    plt.show()


def multivar_normal(mu, cov, n):
    cov = np.array(cov)
    dimensions = np.array(mu).shape[0]
    x = [[normal(0, 1) for x in range(n)] for s in range(dimensions)]

    plt.plot(x[0], x[1], 'x')
    plt.show()


def normal(mu, r):
    return np.random.normal(mu, r)
    y = np.random.rand()
    print(y)
    x = mu + np.sqrt(-(r ** 2) * (np.log(2 * np.math.pi) - 2 * np.log(1 / (r * y))))
    return x


def prob(val, mu, rsq):
    a = 1 / np.sqrt(2 * np.math.pi * rsq)
    p = a * np.exp(-(val * mu) ** 2 / (2 * rsq))
    return p


def plot_pt(val, cls):
    plt.plot(val[0], val[1], 'o', color='green' if cls==1 else 'red')


if __name__ == '__main__':
    # multivar_normal([0,0], [0,0], 5000)
    mu1 = [[0., 0.]]
    mu2 = [[5., 5.]]
    cov1 = [[10., 0.], [0., 10.]]
    cov2 = [[1., 0.], [0., 10.]]
    mu1 = np.array(mu1).T
    mu2 = np.array(mu2).T
    cov1 = np.array(cov1)
    cov2 = np.array(cov2)
    sampled_values = ((np.random.rand(1000, 2,1) - 0.5) * 40)
    samples = []

    delta = 0.125
    x = np.arange(-10.0, 10.0, delta)
    y = np.arange(-10.0, 10.0, delta)
    results = np.zeros((x.shape[0], y.shape[0]))
    X, Y = np.meshgrid(x, y)
    sampled_values = []
    for i in range(len(X)):
        for j in range(len(Y)):
            sampled_values.append([[i, j], [X[i, j], Y[i, j]]])
    for sampled_value in sampled_values:
        i, j = sampled_value[0]
        sampled_value = np.array(sampled_value[1])[np.newaxis].T
        vals = []
        for mu, cov in zip([mu1, mu2], [cov1, cov2]):
            st1 = 1/(2 * np.math.pi * np.sqrt(np.linalg.det(cov)))
            # print('st1', st1)
            # print('(sampled_value - mu).transpose()', (sampled_value - mu).T)
            # print('np.linalg.inv(cov)', np.linalg.inv(cov))
            inner = np.matmul((sampled_value - mu).T, np.linalg.inv(cov))
            # print('inner', inner)
            # print('(sampled_value - mu)', (sampled_value - mu))
            outer = np.matmul(inner, (sampled_value - mu))
            # print('outer', outer)
            val = st1 * np.exp(-0.5 * outer)
            vals.append(val)
        results[i, j] = vals[1] / vals[0]
        samples.append([sampled_value, vals])
    threshold = 1
    mpl(mu1.T[0], cov1, color='yellow')
    mpl(mu2.T[0], cov2, color='orange')
    # [plot_pt(sample, 1 if (val[1] / val[0]) > threshold else 0) for sample, val in samples]
    c = plt.contour(X, Y, results, [10**x for x in np.arange(-2, 3, 0.5)], zorder=1000)
    plt.clabel(c, inline=1, fontsize=5)
    print (results)
    mpl_show()
