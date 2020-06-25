## module import

import numpy as np
import random
import thinkstats2
import thinkplot
import first
from thinkstats2 import Mean, MeanVar, Var, Std, Cov

## function and class definition

def LeastSquares(xs, ys):
    meanx, varx = MeanVar(xs)
    meany = Mean(ys)

    slope = Cov(xs, ys, meanx, meany) / varx
    inter = meany - slope * meanx
    return inter, slope

def FitLine(xs, inter, slope):
    fit_xs = np.sort(xs)
    fit_ys = inter + slope * fit_xs
    return fit_xs, fit_ys

def ScatterFit(xs, ys, **options):
    inter, slope = LeastSquares(xs, ys)
    fit_xs, fit_ys = FitLine(xs, inter, slope)
    thinkplot.Scatter(xs, ys, color='blue', alpha=0.1, s=10)
    thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
    thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
    thinkplot.Show(legend=False, **options)

def Residuals(xs, ys, inter, slope):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    res = ys - (inter + slope * xs)
    return res

def PlotPercentileLines(x_means, cdfs, **options):
    thinkplot.PrePlot(3)
    for percent in [75, 50, 25]:
        y_percentiles = [cdf.Percentile(percent) for cdf in cdfs]
        label = '%dth' % percent
        thinkplot.Plot(x_means, y_percentiles, label=label)

    thinkplot.Show(**options)

def ResampleRows(df):
    """Resamples rows from a DataFrame.

    df: DataFrame

    returns: DataFrame
    """
    return thinkstats2.SampleRows(df, len(df), replace=True)

def SamplingDistributions(live, iters=101):
    t = []
    for _ in range(iters):
        sample = ResampleRows(live)
        ages = sample.agepreg
        weights = sample.totalwgt_lb
        estimates = LeastSquares(ages, weights)
        t.append(estimates)

    inters, slopes = zip(*t)
    return inters, slopes

def SamplingDistributionsWeight(df, iters=101):
    t = []
    for _ in range(iters):
        sample = ResampleRows(df)
        heights = sample.htm3
        weights = sample.wtkg2
        log_weights = np.log10(weights)
        estimates = LeastSquares(heights, log_weights)
        t.append(estimates)

    inters, slopes = zip(*t)
    return inters, slopes

def SamplingMeans(live, col='totalwgt_lb', iters=101):
    means = []
    for _ in range(iters):
        sample = ResampleRows(live)
        data = live[col].values
        means.append(data.mean())

    return means

def Summarize(estimates, actual=None, label=''):
    mean = Mean(estimates)
    stderr = Std(estimates, mu=actual)
    cdf = thinkstats2.Cdf(estimates)
    ci = cdf.ConfidenceInterval(90)
    print(label, 'mean:\n',mean)
    print(label, 'stderr:\n',stderr)
    print(label, 'ci:\n',ci)

def PlotConfidenceIntervals(xs, inters, slopes, percent=90, **options):
    fys_seq = []
    for inter, slope in zip(inters, slopes):
        fxs, fys = FitLine(xs, inter, slope)
        fys_seq.append(fys)

    p = (100-percent) / 2
    percents = p, 100-p
    low, high = thinkstats2.PercentileRows(fys_seq, percents)
    thinkplot.FillBetween(fxs, low, high, **options)

def CoefDetermination(ys, res):
    return 1-Var(res) / Var(ys)

class SlopeTest(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        ages, weights = data
        _, slope = thinkstats2.LeastSquares(ages, weights)
        return slope

    def MakeModel(self):
        _, weights = self.data
        self.ybar = weights.mean()
        self.res = weights - self.ybar

    def RunModel(self):
        ages, _ = self.data
        weights = self.ybar + np.random.permutation(self.res)
        return ages, weights

def ResampleRowsWeighted(df, column='finalwgt'):
    weights = df[column]
    cdf = thinkstats2.Cdf(dict(weights))
    indices = cdf.Sample(len(weights))
    sample = df.loc[indices]
    return sample

## main scripts
if __name__ == '__main__':
    """main scripts"""

    ## make nsfg dataframes
    live, firsts, others = first.MakeFrames()
    live = live.dropna(subset=['agepreg', 'totalwgt_lb'])
    ages = live.agepreg
    weights = live.totalwgt_lb

    ## least squares of birth weight vs. mother's age
    inter, slope = LeastSquares(ages, weights)
    print('inter, slope:\n',inter, slope)

    ## make scatter fit of birth weight vs. mother's age
    ScatterFit(ages, weights)

    ## calculate residuals
    live['residual'] = Residuals(ages, weights, inter, slope)

    ## bin the data and make cdfs
    bins = np.arange(10, 48, 3)
    indices = np.digitize(live.agepreg, bins)
    groups = live.groupby(indices)

    age_means = [group.agepreg.mean() for _, group in groups][1:-1]
    cdfs = [thinkstats2.Cdf(group.residual) for _, group in groups][1:-1]

    ## make plot
    PlotPercentileLines(age_means, cdfs)

    ## assess sampling error of slope inter
    inters, slopes = SamplingDistributions(live, iters=1001)
    ## Summarize
    Summarize(inters, label='intercept')
    Summarize(slopes, label='slope')

    ## Summarize SE and CI for mean birth weight
    means = SamplingMeans(live, col='totalwgt_lb')
    Summarize(means, label='birth weight mean estimate')

    ## plot fit confidence intervals
    PlotConfidenceIntervals(age_means, inters, slopes, percent=90,
                            color='gray', alpha=0.3, label='90% CI')
    PlotConfidenceIntervals(age_means, inters, slopes, percent=50,
                            color='gray', alpha=0.5, label='90% CI')
    thinkplot.Show(xlabel='Mothers age (years)',
                   ylabel='Residual (lbs)',
                   xlim=[10, 45])

    ## goodness of fit
    inter, slope = LeastSquares(ages, weights)
    res = Residuals(ages, weights, inter, slope)
    r2 = CoefDetermination(weights, res)
    print('r2:\n',r2)

    ## use slope test to evaluate correlation between age and birth weight
    ht = SlopeTest((ages, weights))
    pvalue = ht.PValue()
    print('pvalue:\n',pvalue)

    ## plot distribution of Sampling slopes
    inter, slope = LeastSquares(ages, weights)
    inters, slopes = SamplingDistributions(live, iters=1001)
    cdf = thinkstats2.Cdf(slopes)
    thinkplot.Cdf(cdf)
    thinkplot.axvline(slope)
    thinkplot.Show(xlabel='slopes',
                   ylabel='CDF')

    ## weighted resampling
    iters = 100
    estimates = [ResampleRowsWeighted(live).totalwgt_lb.mean()
                 for _ in range(iters)]
    Summarize(estimates, label='means')

    ## non-weighted resampling
    estimates = [ResampleRows(live).totalwgt_lb.mean()
                 for _ in range(iters)]
    Summarize(estimates, label='means')

    #####################################
    ## Evaluate correlation between brfss height and weight
    #####################################
    import brfss

    df = brfss.ReadBrfss(nrows=None)
    df = df.dropna(subset=['htm3', 'wtkg2'])
    heights, weights = df.htm3, df.wtkg2
    log_weights = np.log10(weights)

    ## estimate slope and intercept
    inter, slope = LeastSquares(heights, log_weights)
    print('inter, slope:\n',inter, slope)

    ## make scatter plot
    ScatterFit(heights, log_weights,
           xlabel='height (cm)',
           ylabel='log_weight (log_kg)')

    ## make inverse transform so weights are on linear scale
    fit_xs, fit_ys = FitLine(heights, inter, slope)
    fit_ys_lin = 10**fit_ys
    thinkplot.Scatter(heights, weights, color='blue', alpha=0.1, s=10)
    thinkplot.Plot(fit_xs, fit_ys_lin, color='white', linewidth=3)
    thinkplot.Plot(fit_xs, fit_ys_lin, color='red', linewidth=2)
    thinkplot.Show(legend=False,
                   xlabel='height(cm)',
                   ylabel='weight(kg)')

    ## bin the data and make cdfs
    # calculate residuals
    inter, slope = LeastSquares(heights, log_weights)
    df['residual'] = Residuals(heights, log_weights, inter, slope)

    # bin the data
    bins = np.arange(120, 200, 6)
    indices = np.digitize(df.htm3, bins)
    groups = df.groupby(indices)

    # make cdfs
    height_means = [group.htm3.mean() for _, group in groups][1:-1]
    cdfs = [thinkstats2.Cdf(group.residual) for _, group in groups][1:-1]

    # make plot of percentiles
    PlotPercentileLines(height_means, cdfs,
                        xlabel='height(cm)',
                        ylabel='residual log_10 weight (log_10 kg)')

    ## calculate correlation
    rho = thinkstats2.Corr(heights, log_weights)
    print('rho:\n',rho)

    ## coefficient of determination
    res = df.residual
    r2 = CoefDetermination(log_weights, res)
    print('r2:\n',r2)

    ## confirm that R^2 = rho^2
    print('rho**2:\n',rho**2)
    print('r2:\n',r2)

    ## Std(ys)
    print('Std(log_weights):\n',Std(log_weights))
    print('Std(res):\n',Std(res))
    ratio = 1 - (Std(res) / Std(log_weights))
    print('Knowing patient height reduces RSME by %.1f%%' %ratio)    ## Std(ys)
    print('Std(log_weights):\n',Std(log_weights))
    print('Std(res):\n',Std(res))
    ratio = 1 - (Std(res) / Std(log_weights))
    print('Knowing patient height reduces RSME by {:.1%}'.format(ratio))

    ## make and plot sampling distributions for inter and slope
    inters, slopes = SamplingDistributionsWeight(df, iters=1001)
    ## Summarize
    Summarize(inters, label='intercept')
    Summarize(slopes, label='slope')

    ## plot fit confidence intervals
    x_cdf = thinkstats2.Cdf(heights)
    low, high = x_cdf.Percentile(1), x_cdf.Percentile(99)
    PlotConfidenceIntervals(height_means, inters, slopes, percent=90,
                            color='gray', alpha=0.3, label='90% CI')
    PlotConfidenceIntervals(height_means, inters, slopes, percent=50,
                            color='gray', alpha=0.5, label='90% CI')
    thinkplot.Show(xlabel='Height (cm)',
                   ylabel='Residual (log_10 kg)',
                   xlim=[low, high])

    ## compute p-value of slope
    cdf = thinkstats2.Cdf(slopes)
    pvalue = cdf.Prob(0)
    print('slope pvalue:\n',pvalue)

    ## 90% confidence interval of slope
    ci_90 = cdf.ConfidenceInterval(90)
    print('ci_90:\n',ci_90)

    ## non-weighted resampling
    iters=10
    estimates = [ResampleRows(df).wtkg2.mean() for _ in range(iters)]
    Summarize(estimates, label='mean estimates')

    ## weighted resampling
    estimates = [ResampleRowsWeighted(df,column='finalwt').wtkg2.mean()
                 for _ in range(iters)]
    Summarize(estimates, label='weighted mean estimates')

