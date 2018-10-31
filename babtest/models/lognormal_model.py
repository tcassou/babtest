# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pymc import Lognormal
from pymc import Normal
from pymc import Uniform
from pymc.distributions import lognormal_like

from babtest.models.abstract_model import AbstractModel


class LognormalModel(AbstractModel):

    def __init__(self, control, variant):
        """Init.

        :param np.array control: 1 dimensional array of observations for control group
        :param np.array variant: 1 dimensional array of observations for variant group

        """
        AbstractModel.__init__(self, control, variant)
        self.params = ['location', 'scale']

    def set_models(self):
        """Define models for each group.

        :return: None
        """
        for group in ['control', 'variant']:
            self.stochastics[group] = Lognormal(
                group,
                self.stochastics[group + '_location'],
                self.stochastics[group + '_scale'],
                value=getattr(self, group),
                observed=True)

    def set_priors(self):
        """set parameters prior distributions.

        Hardcoded behavior for now, with non committing prior knowledge.

        :return: None
        """
        obs = np.concatenate((self.control, self.variant))
        mean, sigma, med = np.mean(obs), np.std(obs), np.median(obs)
        location = np.log(med)
        scale = np.sqrt(2 * np.log(mean / med))
        for group in ['control', 'variant']:
            self.stochastics[group + '_location'] = Normal(group + '_location', location, 0.000001 / sigma ** 2)
            self.stochastics[group + '_scale'] = Uniform(group + '_scale', scale / 1000, scale * 1000)

    def draw_distribution(self, group, x, i):
        """Draw the ith sample distribution from the model, and compute its values for each element of x.

        :param string group: specify group, control or variant
        :param numpy.array x: linspace vector, for which to compute probabilities
        :param int i: index of the distribution to compute

        :return: values of the model for the ith distribution
        :rtype: numpy.array
        """
        loc = self.stochastics[group + '_location'].trace()[i]
        scale = self.stochastics[group + '_scale'].trace()[i]
        return np.exp([lognormal_like(xi, loc, scale) for xi in x])
