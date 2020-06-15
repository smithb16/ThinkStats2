## module import and boilerplate

from math import log
import hinc
import scipy.stats
import random
import first
import numpy as np
import thinkplot
import thinkstats2
import matplotlib.pyplot as plt
import analytic

## function definition
def MakeNormalPlot(weights):
    mean = weights.mean()
    std = weights.std()

    xs = [-4, 4]
    fxs, fys = thinkstats2.FitLine(xs, inter=mean, slope=std)
    return fxs, fys

def ExpoVariate(lam):
    """Produce random numbers in an exponential CDF with shape 'lam'
    """
    p = random.random()
    x = -math.log(1-p) / lam
    return x

def ParetoVariate(alpha, xm):
    p = random.random()
    x = (-xm * np.log(alpha)) / np.log(1-p)
    return x

def CMToHeight(cm):
    """Convert height 'cm' to (ft, in)
    return: tuple of (ft, in)
    """
    inch = cm / 2.54
    return divmod(inch, 12)

def MToHeight(m):
    """Convert height 'm' to (ft, in)
    return: tuple of (ft, in)
    """
    return CMToHeight(100*m)

## main scripts
if __name__ == '__main__':
    """main scripts"""
    ## make exponential cdfs
    lambdas = [0.5, 1, 2]
    thinkplot.PrePlot(len(lambdas))
    for lam in lambdas:
        pmf = thinkstats2.MakeExponentialPmf(lam, 3)
        label = r'$\lambda = %g$' % lam
        cdf = thinkstats2.MakeCdfFromPmf(pmf, label=label)
        thinkplot.Cdf(cdf)

    thinkplot.show(xlabel='x', ylabel='CDF')

    ## ReadBabyBoom
    df = analytic.ReadBabyBoom()

    ## plot time differences
    diffs = df.minutes.diff()
    cdf = thinkstats2.Cdf(diffs, label='actual')

    thinkplot.PrePlot(2)
    thinkplot.Cdf(cdf, complement=True)
    pmf = thinkstats2.MakeExponentialPmf(0.0306, 160)
    cdf_model = thinkstats2.MakeCdfFromPmf(pmf)
    thinkplot.Cdf(cdf_model, complement=True, label='approx')
    thinkplot.Show(xlabel='minutes', ylabel='CCDF', yscale='log')

    ## make gaussian cdfs
    params = [(1,0.5), (2,0.4), (3,0.3)]
    thinkplot.PrePlot(len(params))
    for mu, sigma in params:
        label = r'$\mu=%g, \sigma=%g$' %(mu, sigma)
        pmf = thinkstats2.MakeNormalPmf(mu, sigma, 10, n=1000)
        cdf = thinkstats2.MakeCdfFromPmf(pmf, label=label)
        thinkplot.Cdf(cdf, label=label)

    thinkplot.Show(xlabel='x', ylabel='CDF', axis=[-1,4, 0,1])

    ## make data frames
    live, firsts, others = first.MakeFrames()

    ## make normal probability plot of totalwgt_lb
    full_term_wgt = live.loc[live.prglngth >=36, 'totalwgt_lb']
    weights = live.totalwgt_lb
    thinkplot.PrePlot(2)
    fxs, fys = MakeNormalPlot(weights)
    thinkplot.Plot(fxs, fys, color='gray', label='model')

    xs, ys = thinkstats2.NormalProbability(weights)
    thinkplot.Plot(xs, ys, label='all live')

    xs, ys = thinkstats2.NormalProbability(full_term_wgt)
    thinkplot.Plot(xs, ys, label='full term')

    thinkplot.Show(xlabel='standard deviations from mean',
                   ylabel='Birth weight (lbs)',
                   title='Normal probability plot')

    ## make Pareto CDF from random variates
    t = [ParetoVariate(alpha=2, xm=1) for _ in range(1000)]
    cdf = thinkstats2.Cdf(t)
    thinkplot.Cdf(cdf, complement=True)
    thinkplot.Show(xlabel='x', ylabel='CCDF', xscale='log', yscale='log')

    ## what if heights followed Pareto dist?
    pop = 7e9
    alpha = 1.7
    xm = 1

    dist = scipy.stats.pareto(alpha, scale=xm)
    print('median height: %i ft, %i in' %MToHeight(dist.median()))
    mu = dist.mean()
    print('mean height: %i ft, %i in' %MToHeight(mu))
    print('shorter than mean: ', '{:.0%}'.format(dist.cdf(mu)))
    print('taller than 1km: %i' %(dist.sf(1000) * pop))
    tallest = dist.isf(1/pop)
    print('tallest individual: %i m' %tallest)

    ## transform weibull plot into straight line
    sample = [random.weibullvariate(2,1) for _ in range(1000)]
    log_sample = np.log10(sample)
    cdf_log = thinkstats2.Cdf(log_sample, label='sample')
    ys = -1 * np.log(1-cdf_log.ps)

    # weibull plot
    #thinkplot.Plot(-1 * np.log(1-cdf_log.ps))
    thinkplot.Plot(cdf_log.xs, ys)
    thinkplot.Show(xlabel='log10 sample',
                     ylabel='(-ln (CCDF))',
                     yscale = 'log', loc='lower left')

    ## small sample distribution
    df = analytic.ReadBabyBoom()
    diffs = df.minutes.diff()
    cdf = thinkstats2.Cdf(diffs, label='actual')

    n = len(diffs)
    lam = 44.0/24/60
    sample = [random.expovariate(lam) for _ in range(n)]

    print('1/lam:\n', 1/lam)
    print('np.mean(sample):\n', np.mean(sample))

    ## plot simulation vs. data
    cdf_sample = thinkstats2.Cdf(sample, label='simulated')
    thinkplot.PrePlot(2)
    thinkplot.Cdf(cdf, complement=True)
    thinkplot.Cdf(cdf_sample, complement=True)
    thinkplot.Show(xlabel='birth time diffs (min)',
                   ylabel='CDF', yscale='log')

    ## model wealth & income with pareto & lognormal
    df = hinc.ReadData()

    ## assess distribution
    xs, ps = df.income.values, df.ps.values
    cdf = thinkstats2.Cdf(xs, ps, label='data')
    thinkplot.CDFVisualDist(cdf)
