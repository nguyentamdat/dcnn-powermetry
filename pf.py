from filterpy.monte_carlo import systematic_resample
import matplotlib.pyplot as plt
import scipy.stats
from numpy.random import randn
from numpy.linalg import norm
from numpy.random import uniform, randn
import numpy as np
import scipy


def create_uniform_particles(x_range, N):
    particles = np.empty(N)
    particles[:] = uniform(x_range[0], x_range[1], N)
    return particles


def create_gaussian_particles(mean, std, N):
    particles = np.empty(N)
    particles[:] = mean + (randn(N) * std)
    return particles


def predict(particles, u, std):
    N = len(particles)
    particles[:] += u + (randn(N) * std)
    return particles


def update(particles, weights, z, R=2):
    # R = 2
    for i, v in enumerate(particles):
        weights[i] = 1./2./np.pi/R*np.exp(-(v-z)**2/2/R**2)
        if weights[i] > 0.1:
            weights[i] = 0.1
    weights = np.array(weights) / np.sum(weights)
    return weights


def estimate(particles, weights):
    mean = np.average(particles, weights=weights, axis=0)
    var = np.average((particles - mean)**2, weights=weights, axis=0)
    return mean, var


def neff(weights):
    return np.sum(np.square(weights))


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))
    return particles, weights


def run_pf1(N, y, sensors_error=.01, do_plot=True, plot_particles=False, initial_x=None):
    plt.figure(figsize=(16, 10))
    # create particles
    if initial_x:
        particles = create_gaussian_particles(initial_x[0], initial_x[1], N)
    else:
        particles = create_uniform_particles((0, 5), N)
    weights = np.ones(N) / N
    t = range(len(y))
    if do_plot:
        if plot_particles:
            alpha = .20
            if N > 5000:
                alpha *= np.sqrt(5000)/np.sqrt(N)
            plt.scatter([t[0]]*N, particles,
                        alpha=alpha, color='g')

    xs = []
    y = [0] + y
    maee, rmse = 0, 0
    for i, ym in enumerate(y):
        if i == 0:
            continue
        particles = predict(particles, ym - y[i-1], std=2)
        weights = update(particles, weights, ym, R=sensors_error)
        if neff(weights) < N*2 / 3:
            indexes = systematic_resample(weights)
            particles, weights = resample_from_index(
                particles, weights, indexes)
            assert np.allclose(weights, 1/N)
        mu, var = estimate(particles, weights)
        xs.append(mu)
        if do_plot:
            if plot_particles:
                plt.scatter([t[i]]*N, particles, color='k', marker=',', s=1)
            p1 = plt.scatter(t[i], ym, marker='+', color='b', s=18, lw=3)
            p2 = plt.scatter(t[i], mu, marker='+', color='r', s=18, lw=3)
        maee = max(abs(mu - ym), maee)
        rmse += (mu - ym)**2
    if do_plot:
        plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
        plt.grid()
        plt.show()
    print('final position error, variance:\n\t', mu - y[-1], var)
    print('RMSE: {}\nMAEE: {}\n'.format((rmse/(len(y)-1))**(0.5), maee))
    return (rmse/(len(y)-1))**(0.5), maee


def load_data():
    data = np.load("data/batt_pf.npy", allow_pickle=True)
    return data


if __name__ == "__main__":
    a = load_data()
    a = np.array(list(map(lambda x: x[~np.isnan(x)], a)))
    rrmse, rmae = [], []
    for x in a:
        rmse, maee = run_pf1(100, x, plot_particles=True,
                             sensors_error=1, do_plot=False)
        rrmse.append(rmse)
        rmae.append(maee)
    fig, ax = plt.subplots(2, sharex=False, figsize=(12, 8))
    fig.suptitle("Results")
    ax[0].set_ylabel("RMSE", fontsize=14)
    ax[0].bar(range(len(rrmse)), rrmse)
    ax[1].set_ylabel("MAEE", fontsize=14)
    ax[1].bar(range(len(rmae)), rmae)
    fig.show()
    input()
