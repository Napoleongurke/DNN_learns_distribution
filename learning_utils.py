import numpy as np
from scipy.stats import binned_statistic
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import io

def rand_gumbel(lgE, A, size=None, model='EPOS-LHC'):
    """
    Random Xmax values for given energy E [EeV] and mass number A
    See Manlio De Domenico et al., JCAP07(2013)050, doi:10.1088/1475-7516/2013/07/050
    Args:
        lgE (array): energy log10(E/eV)
        A (array): mass number
        model (string): hadronic interaction model
        size (int, optional): number of xmax values to create
    Returns:
        array of random Xmax values in [g/cm^2]
    """
    lE = lgE - 19
    lnA = np.log(A)
    D = np.array([np.ones_like(A), lnA, lnA**2])

    params = {
        'QGSJetII': {
            'mu': ((758.444, -10.692, -1.253), (48.892, 0.02, 0.179), (-2.346, 0.348, -0.086)),
            'sigma': ((39.033, 7.452, -2.176), (4.390, -1.688, 0.170)),
            'lambda': ((0.857, 0.686, -0.040), (0.179, 0.076, -0.0130))},
        'QGSJetII-04': {
            'mu': ((761.383, -11.719, -1.372), (57.344, -1.731, 0.309), (-0.355, 0.273, -0.137)),
            'sigma': ((35.221, 12.335, -2.889), (0.307, -1.147, 0.271)),
            'lambda': ((0.673, 0.694, -0.007), (0.060, -0.019, 0.017))},
        'Sibyll2.1': {
            'mu': ((770.104, -15.873, -0.960), (58.668, -0.124, -0.023), (-1.423, 0.977, -0.191)),
            'sigma': ((31.717, 1.335, -0.601), (-1.912, 0.007, 0.086)),
            'lambda': ((0.683, 0.278, 0.012), (0.008, 0.051, 0.003))},
        'EPOS1.99': {
            'mu': ((780.013, -11.488, -1.906), (61.911, -0.098, 0.038), (-0.405, 0.163, -0.095)),
            'sigma': ((28.853, 8.104, -1.924), (-0.083, -0.961, 0.215)),
            'lambda': ((0.538, 0.524, 0.047), (0.009, 0.023, 0.010))},
        'EPOS-LHC': {
            'mu': ((775.589, -7.047, -2.427), (57.589, -0.743, 0.214), (-0.820, -0.169, -0.027)),
            'sigma': ((29.403, 13.553, -3.154), (0.096, -0.961, 0.150)),
            'lambda': ((0.563, 0.711, 0.058), (0.039, 0.067, -0.004))}}
    param = params[model]

    p0, p1, p2 = np.dot(param['mu'], D)
    mu = p0 + p1 * lE + p2 * lE**2
    p0, p1 = np.dot(param['sigma'], D)
    sigma = p0 + p1 * lE
    p0, p1 = np.dot(param['lambda'], D)
    lambd = p0 + p1 * lE

    return mu - sigma * np.log(np.random.gamma(lambd, 1. / lambd, size=size))


def bootstrap(x, function=np.std, ci=95, iter=1000):
    ''' Add bootsptrapping algorithm. ci=Confidence interval. Note that seaborn regplot uses 95 as default. '''
    def sample(x, function, iter):
        n = len(x)
        vec = []
        for i in range(iter):
            val = np.random.choice(x, n, replace=True)
            vec.append(function(val))
        return np.array(vec)

    def confidence_interval(data, ci):
        low_end = (100 - ci) / 2
        high_end = 100 - low_end
        low_bound = np.percentile(data, low_end)
        high_bound = np.percentile(data, high_end)
        return low_bound, high_bound

    vals = sample(x, function, iter)
    interval = confidence_interval(vals, ci)
    mean = np.mean(vals)
    return mean, interval


def get_performance(x):
    return r"$\mu: %.2f,\; \sigma_{res}: %.2f$" % (np.mean(x), np.std(x))


def plot_performance(y_true, y_pred, name):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), squeeze=True)
    axes = axes.flatten()

    reco = y_pred - y_true

    bins = np.linspace(-3, 6.5, 35)
    # performance
    axes[0].scatter(y_true, y_pred, rasterized=True)
    axes[0].set_xlabel("y_{true}")
    axes[0].set_ylabel("y_{pred}")
    axes[0].set_ylim(-3, 6.5)
    axes[0].text(0.95, 0.075, "corr %.2f" % np.corrcoef(y_pred, y_true)[1, 0], verticalalignment='top',
                 horizontalalignment='right', transform=axes[0].transAxes)

    axes[1].hist(reco, bins, density=True)
    axes[1].set_xlabel("y_{pred} - y_{true}")
    axes[1].text(0.95, 0.95, get_performance(reco), verticalalignment='top', horizontalalignment='right',
                 transform=axes[1].transAxes)

    axes[2].hist(y_pred, label="prediction", bins=bins, alpha=0.5, density=True)
    axes[2].set_xlabel("y_{pred}")

    axes[2].hist(y_true, label="true", bins=bins, alpha=0.5, density=True)
    axes[2].set_xlabel("y_{true}")
    axes[2].legend()

    measured_dispersion, md_err = bootstrap(y_pred, np.std)
    md_err = np.abs(md_err - measured_dispersion)[:, np.newaxis].T
    axes[3].hlines(np.std(y_true), -3, 6.5, label="true width", colors="firebrick")
    axes[3].hlines(np.std(y_pred), -3, 6.5, label="measured width", colors="navy", linestyle="--")
    axes[3].hlines(np.std(reco), -3, 6.5, label="resolution of method", colors="k")
    axes[3].hlines(np.sqrt(np.var(y_pred) - np.var(reco)), -3, 6.5, label="Quadratic corr.", colors="orange", linestyle="-.")

    axes[3].legend(loc="upper right")
    axes[3].set_ylim(0, 4.0)
    axes[3].set_ylabel("width")

    y, _, _ = binned_statistic(y_true, reco, bins=bins)
    x, _, _ = binned_statistic(y_true, y_true, bins=bins)
    axes[4].scatter(y_true, reco, s=0.1, color="grey", rasterized=True)

    axes[4].plot(x, y, color="firebrick")
    axes[4].set_ylabel("y_{pred} - y_{true}")
    axes[4].set_xlabel("y_{true}")
    axes[4].set_ylim(-3, 6.5)
    axes[4].text(0.05, 0.075, "cov %.2f" % np.cov(y_true, reco)[1, 0], verticalalignment='top',
                 horizontalalignment='left', transform=axes[4].transAxes)

    for ax in axes:
        ax.set_xlim(-3, 6.5)

    fig.suptitle(name)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    return fig, axes



def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
