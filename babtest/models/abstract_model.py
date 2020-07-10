# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.lines as mpllines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy.stats
from matplotlib.transforms import blended_transform_factory

PRETTY_BLUE = '#89d1ea'


class AbstractModel(object):

    def __init__(self, control, variant):
        """Init.

        :param np.array control: 1 dimensional array of observations for control group
        :param np.array variant: 1 dimensional array of observations for variant group

        """
        self.control = control
        self.variant = variant
        self.n_samples = None
        self.params = []
        self.stochastics = {}

    def setup(self, n_samples):
        """Define models for each group, set parameters prior distributions and create PyMC Model object.

        :param int n_samples: number of MCMC samples

        :return: None
        """
        self.n_samples = n_samples
        self.set_priors()
        self.set_models()

    def set_models(self):
        """Define models for each group.

        *** To be defined in children classes ***

        :return: None
        """
        raise NotImplementedError

    def set_priors(self):
        """set parameters prior distributions.

        Hardcoded behavior for now, with non committing prior knowledge.

        *** To be defined in children classes ***

        :return: None
        """
        raise NotImplementedError

    def draw_distribution(self, group, x, i):
        """Draw the ith sample distribution from the model, and compute its values for each element of x.

        *** To be defined in children classes ***

        :param string group: specify group, control or variant
        :param numpy.array x: linspace vector, for which to compute probabilities
        :param int i: index of the distribution to compute

        :return: values of the model for the ith distribution
        :rtype: numpy.array
        """
        raise NotImplementedError

    def plot(self, n_bins=30):
        """Generic plot method to display restults of the test.

        :param int n_bins: number of bins in the histograms

        :return: None
        """
        # Observation vs. model
        self.plot_data_and_prediction(n_bins=n_bins)
        # Parameters credibility distributions
        self.plot_params(n_bins=n_bins)
        # Potential extra plots, if defined in children methods
        self.plot_extras(n_bins=n_bins)

    def plot_params(self, n_bins=30):
        """Plot parameters credibility distribution for both groups, and credibility distribution of the difference.

        :param int n_bins: number of bins in the histograms

        :return: None
        """
        f = plt.figure(figsize=(5 * len(self.params), 15), facecolor='white')
        for i, param in enumerate(self.params):
            # Retrieve samples
            control_posterior = self.stochastics['control_' + param].trace()
            variant_posterior = self.stochastics['variant_' + param].trace()
            diff = variant_posterior - control_posterior
            # Plot
            _, edges = np.histogram(np.concatenate((control_posterior, variant_posterior)), bins=n_bins)
            ax1 = f.add_subplot(3, len(self.params), i + 1 + 0 * len(self.params), facecolor='none')
            AbstractModel.plot_posterior(control_posterior, bins=edges, ax=ax1, title='Control ' + param, label='')
            ax2 = f.add_subplot(3, len(self.params), i + 1 + 1 * len(self.params), facecolor='none')
            AbstractModel.plot_posterior(variant_posterior, bins=edges, ax=ax2, title='Variant ' + param, label='')
            ax3 = f.add_subplot(3, len(self.params), i + 1 + 2 * len(self.params), facecolor='none')
            AbstractModel.plot_posterior(diff, bins=n_bins, ax=ax3, title='Diff. of ' + param, draw_zero=True, label='')

    def plot_data_and_prediction(self, n_bins=30, n_curves=10):
        """Compare raw observation distribution with samples of distributions drawned from the model.

        :param int n_bins: number of bins in the histograms
        :param int n_curves: number of sample distributions to draw

        :return: None
        """
        # Observation vs. model
        min_obs = np.minimum(np.min(np.array(self.control)), np.min(np.array(self.variant)))
        max_obs = np.maximum(np.max(np.array(self.control)), np.max(np.array(self.variant)))
        bins = np.linspace(min_obs, max_obs, n_bins)

        f = plt.figure(figsize=(5 * len(self.params), 5), facecolor='white')
        for i, group in enumerate(['control', 'variant']):
            ax = f.add_subplot(2, 1, i + 1, facecolor='none')
            # Observations
            obs = getattr(self, group)
            ax.hist(obs, bins=bins, rwidth=0.5, facecolor='r', edgecolor='none', density=True)

            # Sample of model predictions
            idxs = [int(val) for val in np.round(np.random.uniform(size=n_curves) * self.n_samples)]
            x = np.linspace(bins[0], bins[-1], 100)
            for j in idxs:
                ax.plot(x, self.draw_distribution(group, x, j), color=PRETTY_BLUE, zorder=-10)

            ax.set_xlabel('y')
            ax.set_ylabel('p(y)')
            ax.text(0.8, 0.95, r'$\mathrm{N}_{%s}=%d$' % (group, len(obs)), transform=ax.transAxes, ha='left', va='top')
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
            ax.set_title('{} data vs. posterior prediction'.format(group))

        f.subplots_adjust(hspace=0.5)

    @staticmethod
    def plot_posterior(samples, bins, ax=None, title=None, label='', draw_zero=False):
        """Plot posterior credibility of distribution parameters, for each group and their difference.

        :param numpy.array samples: vector of sample values
        :param numpy.array bins: definition of histogram bins
        :param matplotlib.axes.AxesSubplot ax: subplot object
        :param string title: figure title
        :param string label: label for x-axis
        :param bool draw_zero: whether or not to draw a vertical line in 0

        :return: None
        """
        stats = AbstractModel.samples_statistics(samples)
        ax.set_title(title)
        # Samples distribution
        ax.hist(samples, rwidth=0.8, facecolor=PRETTY_BLUE, edgecolor='none', bins=bins)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(stats['mean'], 0.99, '%s = %.3g' % ('mean', stats['mean']), transform=trans, ha='center', va='top')
        ax.text(stats['mean'], 0.95, '%s = %.3g' % ('mode', stats['mode']), transform=trans, ha='center', va='top')
        # Vertical line in 0
        if draw_zero:
            ax.axvline(0, linestyle=':', color='r', linewidth=2)
        # Highest Density Interval (HDI)
        hdi_line, = ax.plot([stats['hdi_min'], stats['hdi_max']], [0, 0], lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text(stats['hdi_min'], 0.04, '%.3g' % stats['hdi_min'], transform=trans, ha='center', va='bottom')
        ax.text(stats['hdi_max'], 0.04, '%.3g' % stats['hdi_max'], transform=trans, ha='center', va='bottom')
        ax.text((stats['hdi_min'] + stats['hdi_max']) / 2, 0.14, '95% HDI', transform=trans, ha='center', va='bottom')
        # Display option
        ax.spines['bottom'].set_position(('outward', 2))
        for loc in ['left', 'top', 'right']:
            ax.spines[loc].set_color('none')        # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([])                      # don't draw
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        ax.set_xlabel(label)

    def plot_extras(self, n_bins=30):
        """If overloaded in children classes, add extra plots to the display.

        :param int n_bins: number of bins in the histograms

        :return: None
        """

    @staticmethod
    def hdi(samples, cred_mass=0.95):
        """Compute bounds of Highest Density Interval.

        :param numpy.array samples: vector of sample values
        :param float cred_mass: target ratio of values in the Highest Density Interval

        :return: min and max bounds of the interval
        :rtype: (float)
        """
        sorted_samples = np.sort(samples)
        interval_size = int(np.floor(cred_mass * samples.size))
        interval_width = sorted_samples[interval_size:] - sorted_samples[:samples.size - interval_size]
        min_idx = np.argmin(interval_width)
        hdi_min = sorted_samples[min_idx]
        hdi_max = sorted_samples[min_idx + interval_size]
        return hdi_min, hdi_max

    @staticmethod
    def samples_statistics(samples):
        """Compute basic statistics over an input set of values.

        :param numpy.array samples: vector of sample values

        :return: dict of statistics
        :rtype: {string: float}
        """
        hdi_min, hdi_max = AbstractModel.hdi(samples)
        mean_val = np.mean(samples)
        # Calculate mode (use kernel density estimate)
        kernel = scipy.stats.gaussian_kde(samples)
        bw = kernel.covariance_factor()
        x = np.linspace(np.min(samples) - 3 * bw ** 2, np.max(samples) + 3 * bw ** 2, 512)
        vals = kernel.evaluate(x)
        mode_val = x[np.argmax(vals)]
        return {
            'hdi_min': hdi_min,
            'hdi_max': hdi_max,
            'mean': mean_val,
            'mode': mode_val,
        }
