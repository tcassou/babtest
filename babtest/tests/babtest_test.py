# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np
from genty import genty
from genty import genty_dataset
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from babtest.babtest import BABTest


@genty
class BABTestTest(unittest.TestCase):

    @genty_dataset(
        (np.random.normal, (0, 1, 55), (0.1, 1.05, 62), 'gaussian'),
        (np.random.exponential, (1, 100), (1.1, 102), 'exponential'),
        (np.random.lognormal, (0, 1.0, 10), (0, 1.1, 12), 'lognormal'),
        (np.random.randint, (0, 2, 10), (0, 2, 12), 'bernoulli'),
        (np.random.normal, (0, 1.0, 10), (0, 1.1, 12), 'student'),
        (np.random.poisson, (1.0, 77), (2.1, 86), 'poisson'),
    )
    def test_run(self, dist, param1, param2, model):
        y1 = dist(*param1)
        y2 = dist(*param2)
        bt = BABTest(y1, y2, model=model, verbose=False)
        bt.run(n_iter=1100, n_burn=100)
        assert_array_equal(bt.control, y1)
        assert_array_equal(bt.variant, y2)
        assert_array_equal(bt.model.control, y1)
        assert_array_equal(bt.model.variant, y2)
        self.assertTrue('control' in bt.model.stochastics)
        self.assertTrue('variant' in bt.model.stochastics)
        assert_array_equal(bt.model.stochastics['control'].get_value(), y1)
        assert_array_equal(bt.model.stochastics['variant'].get_value(), y2)

    def test_unknown_model(self):
        y1 = np.array([0])
        y2 = np.array([1])
        assert_raises(KeyError, BABTest, y1, y2, model='foo')

    @genty_dataset(
        (150, 15),
        (150, 0),
        (150, 149),
    )
    def test_sampling(self, n_iter, n_burn):
        y1 = np.random.randint(0, 2, 10)
        y2 = np.random.randint(0, 2, 11)
        bt = BABTest(y1, y2, model='bernoulli', verbose=False)
        bt.run(n_iter=n_iter, n_burn=n_burn)
        assert_array_equal(bt.control, y1)
        assert_array_equal(bt.variant, y2)
        assert_array_equal(bt.model.control, y1)
        assert_array_equal(bt.model.variant, y2)
        self.assertTrue('control' in bt.model.stochastics)
        self.assertTrue('variant' in bt.model.stochastics)
        assert_array_equal(bt.model.stochastics['control'].get_value(), y1)
        assert_array_equal(bt.model.stochastics['variant'].get_value(), y2)
        self.assertEqual(bt.model.stochastics['control_p'].trace().shape, (n_iter - n_burn,))
        self.assertEqual(bt.model.stochastics['variant_p'].trace().shape, (n_iter - n_burn,))
