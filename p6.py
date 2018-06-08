import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

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
    return np.array([val, idx])


colors = np.array(['b', 'g', 'r', 'y', 'c', 'm'])
# np.random.shuffle(colors)

shapes = ['o', 's', 'x']


def plot_pt(val, cls, actual):
    # print(colors[cls], cls)
    plt.plot(val[0], val[1], shapes[actual], color=colors[cls])


data = np.array([gmm() for x in range(100)])
pts = data[:, 0]
classes = data[:, 1]

# initial guesses - intentionally bad
guess = {'lambda': []}
for m in range(n):
    guess['mu' + str(m)] = np.array(mus[m])
    guess['cov' + str(m)] = np.array(covs[m])
    guess['lambda'].append(1 / float(n))


def normal_pdf(v, mean, rsq):
    a = 1 / np.sqrt(2 * np.math.pi * rsq)
    p = a * np.exp(-(v - mean) ** 2 / (2 * rsq))
    return p


def prob(val, mu, sig, lam):
    p = lam
    for i in range(len(val)):
        p *= norm.pdf(val[i], mu[i], sig[i][i])
    return p


def expectation(pts, guess):
    results = []
    for i in range(pts.shape[0]):
        x, y = pts[i]
        probs = [prob([x, y], guess['mu' + str(m)], guess['cov' + str(m)], guess['lambda'][m]) for m in range(n)]
        results.append(np.argmax(probs))
    return np.array(results)


def maximization(ps, results, g):
    params = g.copy()
    for m in range(n):
        this_class = np.array(ps[results == m].tolist())
        if this_class.shape[0] == 0:
            continue
        x = this_class[:, 0]
        y = this_class[:, 1]
        params['mu' + str(m)] = [x.mean(), y.mean()]
        params['cov' + str(m)] = [
            [x.std(), 0],
            [0, y.std()]
        ]
        params['lambda'][m] = this_class.shape[0] / ps.shape[0]
    return params

def distance(old_params, new_params):
    dist = 0
    for param in ['mu0', 'mu1', 'mu2']:
        print (param)
        for i in range(len(old_params[param])):
            dist += (old_params[param][i] - new_params[param][i])**2
    return dist ** 0.5


shift = 10
epsilon = 0.01
iters = 0
true_params = guess.copy()
updated_params = guess.copy()
distances = []
while iters < 10:  # shift > epsilon:
    iters += 1
    shift = distance(true_params, updated_params)
    distances.append(shift)

    updated_lbls = expectation(pts, updated_params)
    updated_params = maximization(pts, updated_lbls, updated_params)


    print(f'iteration {iters}, shift {shift}')

[plot_pt(p, c, actual) for p, c, actual in zip(pts, updated_lbls, classes)]

plt.show()

    # time.sleep(0.25)
plt.close('all')
plt.plot(np.arange(len(distances)), distances)
plt.title('Distance')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.show()
print (guess, true_params)