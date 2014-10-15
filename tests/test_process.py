# -*- coding: utf-8 -*-
import unittest as ut

import numpy as np
import pandas as pd
import pandas.util.testing as pdt
import collections as co
import itertools as it


import process as pr
pr.SEP = ','
pr.REFSET = 'S0'
pr.REFSFX = 'a'

# ------------------------------------------------------------------------------

class Test_base(ut.TestCase):
    def setUp(self):
        self.sep = sep = ','
        colnames = (['id', 'x'] +
                    [sep.join(p) for p in
                     it.product(['S%d' % i for i in 1, 0, 2], 'abc')])
        index = ['C', 'A', 'B']

        self.data = table_(
             [['C', 'd',  40,  22, 192,  10,  18, 100,  20,  12,   4],
              ['A', 'e',  60,   0,  12,  20,  12,  56,   0,   5,   4],
              ['B', 'f',  20,   8,  36,  30,  90,  84,  10,  13,  52]],
             colnames,
             index=index)

        self.norm_a = table_(
             [['C', 'd',  40,  88,  96,  10,   9,  25,  20,  12,   2],
              ['A', 'e',  60,   0,   6,  20,   6,  14,   0,   5,   2],
              ['B', 'f',  20,  32,  18,  30,  45,  21,  10,  13,  26]],
             colnames,
             index=index)

        self.norm_ab = table_(
             [['C', 'd',  20,  44,  48,  10,   9,  25,  40,  24,   4],
              ['A', 'e',  30,   0,   3,  20,   6,  14,   0,  10,   4],
              ['B', 'f',  10,  16,   9,  30,  45,  21,  20,  26,  52]],
             colnames,
             index=index)

        inf = np.inf
        self.norm_abc = table_(
             [['C', 'd',  10,  22,  24,  10,   9,  25,  10,   6,   1],
              ['A', 'e',  20,   0,   2,  20,   6,  14,   0, inf, inf],
              ['B', 'f',  30,  48,  27,  30,  45,  21,  30,  39,  78]],
             colnames,
             index=index)

    def assertAlmostEqual(self, x, y):
        if isinstance(x, np.ndarray):
            if (np.all(~(np.isinf(x) ^ np.isinf(y))) and
                np.all(~(np.isnan(x) ^ np.isnan(y)))):
                x = np.nanmax(np.abs(x - y))
                y = 0
            else:
                assert False
        return super(Test_base, self).assertAlmostEqual(x, y)

# ------------------------------------------------------------------------------

class Test_happypath(Test_base):
    def assertEqual(self, first, second, msg=None):
        if isinstance(first, pd.DataFrame):
            return pdt.assert_frame_equal(first, second,
                                          check_index_type=True,
                                          check_column_type=True,
                                          check_frame_type=True)

        if isinstance(first, pd.Series):
            return pdt.assert_series_equal(first, second,
                                           check_index_type=True,
                                           check_series_type=True)

        if isinstance(first, np.ndarray):
            return self.assertTrue((first == second).all())

        return super(Test_base, self).assertEqual(first, second, msg)

    def test_a(self):
        actual = pr.normalize_a(self.data)
        expected = self.norm_a
        self.assertAlmostEqual(actual.iloc[:, 2:].values, expected.iloc[:, 2:].values)

    def test_b(self):
        actual = pr.normalize_b(self.norm_a)
        expected = self.norm_ab
        self.assertAlmostEqual(actual.iloc[:, 2:].values, expected.iloc[:, 2:].values)

    def test_ab(self):
        actual = pr.normalize_ab(self.data)
        expected = self.norm_ab
        self.assertAlmostEqual(actual.iloc[:, 2:].values, expected.iloc[:, 2:].values)

    def test_c(self):
        actual = pr.normalize_c(self.norm_ab)
        expected = self.norm_abc
        self.assertAlmostEqual(actual.iloc[:, 2:].values, expected.iloc[:, 2:].values)

# ------------------------------------------------------------------------------

def table_(data, varnames, **kwargs):
    assert len(data) > 0
    assert len(set([len(varnames)] + map(len, data))) == 1
    return pd.DataFrame(co.OrderedDict([(varnames[i], [row[i] for row in data])
                                        for i in range(len(varnames))]),
                        **kwargs)

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import test.test_support as ts
    ts.run_unittest(Test_happypath)
