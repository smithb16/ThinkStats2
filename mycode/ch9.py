## module import and boilerplate
import numpy as np
import random
import thinkstats2
import thinkplot
import first
import nsfg2

## function and class definition

class HypothesisTest(object):

    def __init__(self, data):
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)

    def PValue(self, iters=1000):
        self.test_stats = [self.TestStatistic(self.RunModel())
                            for _ in range(iters)]

        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def TestStatistic(self, data):
        raise UnimplementedMethodException()

    def MakeModel(self):
        pass

    def RunModel(self):
        raise UnimplementedMethodException()


class CoinTest(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        heads, tails = data
        test_stat = abs(heads-tails)
        return test_stat

    def RunModel(self):
        heads, tails = self.data
        n = heads + tails
        sample = [random.choice('HT') for _ in range(n)]
        hist = thinkstats2.Hist(sample)
        data = hist['H'], hist['T']
        return data

class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[ : self.n], self.pool[ self.n : ]
        return data

class DiffMeansResample(DiffMeansPermute):

    def RunModel(self):
        sample1 = np.random.choice(self.pool, size=self.n, replace=True)
        sample2 = np.random.choice(self.pool, size=self.m, replace=True)
        return sample1, sample2

class DiffMeansOneSided(DiffMeansPermute):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.mean() - group2.mean()
        return test_stat

class DiffStdPermute(DiffMeansPermute):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.std() - group2.std()
        return test_stat

class CorrelationPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys

class DiceTest(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        observed = data
        n = sum(observed)
        expected = np.ones(6) * n / 6
        test_stat = sum(abs(observed - expected))
        return test_stat

    def RunModel(self):
        n = sum(self.data)
        values = [1,2,3,4,5,6]
        rolls = np.random.choice(values, n, replace=True)
        hist = thinkstats2.Hist(rolls)
        freqs = hist.Freqs(values)
        return freqs

class DiceChiTest(DiceTest):

    def TestStatistic(self, data):
        observed = data
        n = sum(observed)
        expected = np.ones(6) * n / 6
        test_stat = sum((observed - expected) ** 2 / expected)
        return test_stat

class PregLengthTest(thinkstats2.HypothesisTest):

    def MakeModel(self):
        firsts, others = self.data
        self.n = len(firsts)
        self.pool = np.hstack((firsts, others))

        pmf = thinkstats2.Pmf(self.pool)
        self.values = range(35, 44)
        self.expected_probs = np.array(pmf.Probs(self.values))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[ : self.n], self.pool[self.n : ]
        return data

    def TestStatistic(self, data):
        firsts, others = data
        stat = self.ChiSquared(firsts) + self.ChiSquared(others)
        return stat

    def ChiSquared(self, lengths):
        hist = thinkstats2.Hist(lengths)
        observed = np.array(hist.Freqs(self.values))
        expected = self.expected_probs * len(lengths)
        stat = sum((observed - expected)**2 / expected)
        return stat

def FalseNegRate(data, num_runs=1000):
    """Computes the chance of a false negative based on resampling.

    data: pair of sequences
    num_runs: how many experiments to simulate

    return: float false negative rate
    """
    group1, group2 = data
    count = 0

    for _ in range(num_runs):
        sample1 = thinkstats2.Resample(group1)
        sample2 = thinkstats2.Resample(group2)
        ht = DiffMeansPermute((sample1, sample2))
        pvalue = ht.PValue(iters=101)
        if pvalue > 0.05:
            count += 1

    return count / num_runs

## main scripts
if __name__ == '__main__':
    """main scripts"""

    ## test for coin bias with 140 heads and 110 tails on 250 flips
    ct = CoinTest((140, 110))
    pvalue = ct.PValue()
    print('pvalue:\n',pvalue)

    ## make nsfg data frames
    live, firsts, others = first.MakeFrames()
    data_length = firsts.prglngth.values, others.prglngth.values
    firsts = firsts.dropna(subset=['totalwgt_lb'])
    others = others.dropna(subset=['totalwgt_lb'])
    data_weight = firsts.totalwgt_lb.values, others.totalwgt_lb.values

    ## test difference in pregnancy lengths
    ht = DiffMeansPermute(data_length)
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)
    ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF')

    ## test difference in pregnancy weights
    ht = DiffMeansPermute(data_weight)
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)
    ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF')

    ## test difference in pregnancy lengths with one-sided hypothesis
    ht = DiffMeansOneSided(data_length)
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)
    ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF')

    ## test difference in pregnancy length standard deviation between firsts and others
    ht = DiffStdPermute(data_length)
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)
    ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF')

    ## test correlation between age and birth weight
    cleaned = live.dropna(subset=['agepreg', 'totalwgt_lb'])
    data = cleaned.agepreg.values, cleaned.totalwgt_lb.values
    ht = CorrelationPermute(data)
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)

    ## test frequencies of die
    data = [8,9,19,5,8,11]
    data2 = [2*val for val in data]
    dt = DiceTest(data2)
    pvalue = dt.PValue(iters=1000)
    print('pvalue:\n',pvalue)

    ## test dice with chi-squared
    data = [8,9,19,5,8,11]
    dt = DiceChiTest(data)
    pvalue = dt.PValue(iters=1000)
    print('pvalue:\n',pvalue)

    ## test pregnancy lengths from 35-43 weeks using chi squared
    data = firsts.prglngth.values, others.prglngth.values
    ht = PregLengthTest(data)
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)
    print('ht.actual:\n',ht.actual)
    print('ht.MaxTestStat():\n',ht.MaxTestStat())

    ## find FalseNegRate of pregnancy length data
    data = firsts.prglngth.values, others.prglngth.values
    neg_rate = FalseNegRate(data)
    print('neg_rate:\n',neg_rate)


    #################################
    ## Test hypotheses with nsfg2 data
    #################################
    live, firsts, others = nsfg2.MakeFrames()

    ## test difference in pregnancy lengths
    data_length = firsts.prglngth.values, others.prglngth.values
    ht = DiffMeansPermute(data_length)
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)
    ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF')

    ## evaluate p value vs. sample size
    ns = np.logspace(2, 12, num=50, base=2, dtype=int)
    pvalues = []

    for n in ns:
        firsts_sub = thinkstats2.SampleRows(firsts, n)
        others_sub = thinkstats2.SampleRows(others, n)
        data = firsts_sub.prglngth.values, others_sub.prglngth.values

        ht = DiffMeansPermute(data)
        pvalue = ht.PValue()
        pvalues.append(pvalue)
        print('sample size: %d \npvalue: %f\n' %(n, pvalue))

    thinkplot.Plot(ns, pvalues,'.')
    thinkplot.axhline(0.05)
    thinkplot.Show(xlabel='sample size',
                   ylabel='p values')

    ## test difference in pregnancy lengths with resampling
    data_length = firsts.prglngth.values, others.prglngth.values
    ht = DiffMeansResample(data_length)
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)
    ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF')

    ## test difference in birth weight with resampling
    ht = DiffMeansResample(data_weight)
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)
    ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF')
