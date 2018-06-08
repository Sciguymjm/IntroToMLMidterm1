import matplotlib.pyplot as plt
import numpy as np

n = 3
p = np.random.randint(1, 10, n)
p = p / np.sum(p)
print(p)

mus = []
covs = []
dimensions = 2
for m in range(n):
    mean = [np.random.randint(0, 10) + (m * 5) for d1 in range(dimensions)]
    cov = [
        [[np.random.randint(1, 10), 0],
         [0, np.random.randint(1, 10)]]
        for d2 in range(dimensions)
    ]
    mus.append(mean)
    covs.append(cov[0])


def gmm():
    idx = np.random.choice(np.arange(0, n), p=p)
    mean = mus[idx]
    cov = covs[idx]
    val = np.random.multivariate_normal(mean, cov)
    return val, idx


colors = ['b', 'g', 'r', 'y']


def plot_pt(val, cls):
    print(colors[cls], cls)
    plt.plot(val[0], val[1], 'o', color=colors[cls])


data = np.array([gmm() for x in range(1000)])
pts = data[:, 0]
classes = data[:, 1]

[plot_pt(p, c) for p, c in data]

plt.show()
