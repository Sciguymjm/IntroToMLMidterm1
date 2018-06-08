import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg.decomp import eigh
from scipy.special import erfinv


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
    rand = np.random.rand()
    x = np.sqrt(2) * r * erfinv(2 * rand - 1) + mu
    return x
    y = np.random.rand()
    print(y)
    x = mu + np.sqrt(-(r ** 2) * (np.log(2 * np.math.pi) - 2 * np.log(1 / (r * y))))
    return x


def inv_erf(z):
    def c_k(k):
        if k < 2:
            return 1
        return np.sum([(c_k(m) * c_k(k - 1 - m) / ((m + 1) * (2 * m + 1))) for m in range(0, k - 1)])

    return np.sum([c_k(k) / (2 * k + 1) * (np.sqrt(np.math.pi / 2 * z)) ** (2 * k + 1) for k in range(0, 10)])


def test_custom_normal():
    data = [normal(0, 1) for x in range(10000)]
    print('std', np.std(data), 'mean', np.mean(data))
    # plt.hist(data, bins=20)
    # plt.show()


def numpy_normal(mu, r):
    return np.random.normal(mu, r)


def smthn(val, mu, rsq):
    a = 1 / np.sqrt(2 * np.math.pi * rsq)
    p = a * np.exp(-(val * mu) ** 2 / (2 * rsq))
    return p


def plot_pt(val, cls):
    plt.plot(val[0], val[1], 'o', color='green' if cls == 1 else 'red')


def plot_obs(val, cls, accurate):
    plt.plot(val[0], val[1], 'o' if cls == 0 else 's', color='green' if accurate else 'red')


mus = np.array([
    [[0., 0.]],
    [[5., 5.]]
])
covs = np.array([
    [[5., 0.], [0., 10.]],
    [[1., 0.], [0., 10.]]
])

n = 2
prob = [0.5, 0.5]


def generate_sample_gmm() -> (np.ndarray, int):
    idx = np.arange(n)
    choice = np.random.choice(idx, p=prob)
    return np.random.multivariate_normal(mus[choice][0], covs[choice]), choice


def pdf(sampled_value: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    st1 = 1 / (2 * np.math.pi * np.sqrt(np.linalg.det(cov)))
    # print('st1', st1)
    # print('(sampled_value - mu).transpose()', (sampled_value - mu).T)
    # print('np.linalg.inv(cov)', np.linalg.inv(cov))
    inner = np.matmul((sampled_value - mu).T, np.linalg.inv(cov))
    # print('inner', inner)
    # print('(sampled_value - mu)', (sampled_value - mu))
    outer = np.matmul(inner, (sampled_value - mu))
    # print('outer', outer)
    val = st1 * np.exp(-0.5 * outer)
    return val


if __name__ == '__main__':
    print(generate_sample_gmm())
    test_custom_normal()
    # multivar_normal([0,0], [0,0], 5000)
    # QDA #
    mus_t = [np.array(m).T for m in mus]
    covs_t = [np.array(c) for c in covs]
    sampled_values = ((np.random.rand(1000, 2, 1) - 0.5) * 40)
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
        for mu, cov in zip(mus_t, covs_t):
            val = pdf(sampled_value, mu, cov)
            vals.append(val)
        results[i, j] = vals[1] / vals[0]
        samples.append([sampled_value, vals])
    threshold = 1
    mpl(mus_t[0].T[0], covs_t[0], color='yellow')
    mpl(mus_t[1].T[0], covs_t[1], color='orange')
    # [plot_pt(sample, 1 if (val[1] / val[0]) > threshold else 0) for sample, val in samples]
    c = plt.contour(X, Y, results, [1], zorder=1000)
    # c = plt.contour(X, Y, results, [10 ** x for x in np.arange(-2, 3, 0.5)], zorder=1000)
    plt.clabel(c, inline=1, fontsize=5)

    sample_pts = np.array([generate_sample_gmm() for x in range(100)])
    pts = sample_pts[:, 0]
    labels = sample_pts[:, 1]

    vals = []
    for pt in pts:
        pdfs = [pdf(pt[np.newaxis].T, mu, cov) for mu, cov in zip(mus_t, covs_t)]
        vals.append(int((pdfs[1] / pdfs[0])[0] > 1))
    correct = np.array(labels) == np.array(vals)
    [plot_obs(pt, v, c) for pt, v, c in zip(pts, labels, correct)]
    acc = correct.sum() / len(labels)
    print('acc', acc)
    # print(results)

    mpl_show()

    # LDA #

    Sb = (mus[0] - mus[1]) * (mus[0] - mus[1]).T
    Sw = covs[0] + covs[1]
    D, V = eigh(Sb, Sw)
    D = np.stack(([0., 0.], D))
    # assert (np.matmul(Sb, V) == np.matmul(np.matmul(W, Sw), V)).all()
    ind = np.argmax(np.diag(D))
    w = V[:, ind]
    print(w)
    x1 = np.array(pts[labels == 0].tolist())
    x2 = np.array(pts[labels == 1].tolist())
    y1 = np.matmul(w, x1.T)
    y2 = np.matmul(w, x2.T)


    plt.plot(np.arange(len(y1)), y1, 'x', color='b')
    plt.plot(np.arange(len(y2)), y2, 'o', color='r')
    plt.show()

    # threshold = np.arange(0, 10, 1)
    # current_min = 1e10
    # current_min_t = -1
    # for t in threshold:
    #     error_rate = lda_error_rate(threshold)
    #     if error_rate < current_min:
    #         current_min = error_rate
    #         current_min_t = t
    # print(current_min_t, current_min)
    # plt.plot()
