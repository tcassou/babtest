# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from pymc import deterministic
from pymc import Exponential
from pymc import NoncentralT
from pymc import Normal
from pymc import Uniform
from pymc.distributions import noncentral_t_like

from babtest.models.abstract_model import AbstractModel


class StudentModel(AbstractModel):

    def __init__(self, control, variant):
        """Init.

        :param np.array control: 1 dimensional array of observations for control group
        :param np.array variant: 1 dimensional array of observations for variant group

        """
        AbstractModel.__init__(self, control, variant)
        self.params = ['mean', 'sigma', 'nu_minus_one']

    def set_models(self):
        """Define models for each group.

        :return: None
        """
        def get_nu(group):
            @deterministic(plot=False)
            def nu(nu_minus_one=self.stochastics[group + '_nu_minus_one']):
                return nu_minus_one + 1
            return nu

        def get_lam(group):
            @deterministic(plot=False)
            def lam(sigma=self.stochastics[group + '_sigma']):
                return 1 / sigma ** 2
            return lam

        for group in ['control', 'variant']:
            self.stochastics[group] = NoncentralT(
                group,
                self.stochastics[group + '_mean'],
                get_lam(group),
                get_nu(group),
                value=getattr(self, group),
                observed=True)

    def set_priors(self):
        """set parameters prior distributions.

        Hardcoded behavior for now, with non committing prior knowledge.

        :return: None
        """
        obs = np.concatenate((self.control, self.variant))
        obs_mean, obs_sigma = np.mean(obs), np.std(obs)
        for group in ['control', 'variant']:
            self.stochastics[group + '_mean'] = Normal(group + '_mean', obs_mean, 0.000001 / obs_sigma ** 2)
            self.stochastics[group + '_sigma'] = Uniform(group + '_sigma', obs_sigma / 1000, obs_sigma * 1000)
            self.stochastics[group + '_nu_minus_one'] = Exponential(group + '_nu_minus_one', 1 / 29)

    def draw_distribution(self, group, x, i):
        """Draw the ith sample distribution from the model, and compute its values for each element of x.

        :param string group: specify group, control or variant
        :param numpy.array x: linspace vector, for which to compute probabilities
        :param int i: index of the distribution to compute

        :return: values of the model for the ith distribution
        :rtype: numpy.array
        """
        m = self.stochastics[group + '_mean'].trace()[i]
        s = self.stochastics[group + '_sigma'].trace()[i]
        nu = self.stochastics[group + '_nu_minus_one'].trace()[i] + 1
        lam = 1 / s ** 2
        return np.exp([noncentral_t_like(xi, m, lam, nu) for xi in x])

    def plot_extras(self, n_bins=30):
        """Adding to the parent plot the display of effect size.

        :param int n_bins: number of bins in the histograms

        :return: None
        """
        diff_means = self.stochastics['variant_mean'].trace() - self.stochastics['control_mean'].trace()
        avg_std = np.sqrt(
            (self.stochastics['control_sigma'].trace() ** 2 + self.stochastics['variant_sigma'].trace() ** 2) / 2)
        f = plt.figure(figsize=(5 * len(self.params), 5), facecolor='white')
        ax = f.add_subplot(1, 1, 1, facecolor='none')
        AbstractModel.plot_posterior(
            diff_means / avg_std, bins=n_bins, ax=ax, title='Effect Size',
            draw_zero=True, label=r'$(\mu_1 - \mu_2)/\sqrt{(\sigma_1 + \sigma_2)/2}$')
