## module import and boilerplate

import pandas as pd
import numpy as np
import brfss
import thinkstats2
import thinkplot

## function and class def

def SpearmanCorr(xs, ys):
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return Corr(xranks, yranks)

def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr

def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov

def SampleRows(df, nrows, replace=False):
    """Creates sample of size 'nrows' from DataFrame 'df'
    attr:
        df: pandas DataFrame
        nrows: int number of samples
        replace: boolean to replace after selection
    returns: pandas.DataFrame
    """
    indices = np.random.choice(df.index, nrows, replace=replace)
    sample = df.loc[indices]
    return sample

def Jitter(values, jitter=0.5):
    B
    n = len(values)
    return np.random.normal(0, jitter, n) + values

## Main scripts
if __name__ == '__main__':
    # create brfss DataFrame
    df = brfss.ReadBrfss(nrows=None)
    sample = SampleRows(df, 5000)
    heights, weights = sample.htm3, sample.wtkg2

    ## Jitter data
    heights = Jitter(heights, 1.4)
    weights = Jitter(weights, 0.5)

    ## make scatter plot
    thinkplot.Scatter(heights, weights, alpha=0.1, s=10)
    thinkplot.Show(xlabel='Height (cm)',
                   ylabel='Weight (kg)',
                   axis = [140, 210, 20, 200],
                   legend=False)

    ## make HexBin plot
    thinkplot.HexBin(heights, weights)
    thinkplot.Show(xlabel='Height (cm)',
                   ylabel='Weight (kg)',
                   axis = [140, 210, 20, 200],
                   legend=False)

    ## now use entire dataset
    heights_all, weights_all = sample.htm3, sample.wtkg2
    heights_all = Jitter(heights, 1.4)
    weights_all = Jitter(weights, 0.5)

    ## make scatter plot
    thinkplot.PrePlot(num=2, cols=2)
    thinkplot.SubPlot(1)
    thinkplot.Scatter(heights_all, weights_all, alpha=0.1, s=10)
    thinkplot.Show(xlabel='Height (cm)',
                   ylabel='Weight (kg)',
                   axis = [140, 210, 20, 200],
                   legend=False)

    thinkplot.SubPlot(2)
    thinkplot.HexBin(heights_all, weights_all)
    thinkplot.Show(xlabel='Height (cm)',
                   ylabel='Weight (kg)',
                   axis = [140, 210, 20, 200],
                   legend=False)

    ## bin data
    cleaned = df.dropna(subset=['htm3','wtkg2'])
    bins = np.arange(135, 210, 5)
    indices = np.digitize(cleaned.htm3, bins)
    groups = cleaned.groupby(indices)

    ## print binned data
    for i, group in groups:
        print(i, len(group))

    ## compute cdf for each group
    mean_heights = [group.htm3.mean() for i, group in groups]
    cdfs = [thinkstats2.Cdf(group.wtkg2) for i, group in groups]

    ## extract 25th, 50th, 75th percentiles
    for percent in [75, 50, 25]:
        weight_percentiles = [cdf.Percentile(percent) for cdf in cdfs]
        label = '%dth' %percent
        thinkplot.Plot(mean_heights, weight_percentiles, label=label)

    thinkplot.Show(xlabel='Height (cm)',
                   ylabel='Weight (kg)',
                   axis = [140, 210, 20, 200],
                   legend=False)

    ## re-bin data and make new cdfs
    bins = np.arange(135, 210, 15)
    indices = np.digitize(cleaned.htm3, bins)
    groups = cleaned.groupby(indices)
    mean_heights = [group.htm3.mean() for i, group in groups]
    cdfs = [thinkstats2.Cdf(group.wtkg2) for i, group in groups]

    ## plot the cdfs
    thinkplot.PrePlot(len(cdfs))
    for i, cdf in enumerate(cdfs):
        if i==0:
            label='<%d cm' %bins[0]
        elif i==len(cdfs)-1:
            label='>%d cm' %bins[-1]
        else:
            label = '%d - %d cm' % (bins[i-1], bins[i])
        thinkplot.Cdf(cdf, label=label)
    thinkplot.Show(xlabel='weight (kg)',
                   ylabel='CDF',
                   legend=True)

    ## calculate covariance of height and weight
    heights, weights = cleaned.htm3, cleaned.wtkg2
    print('Cov(heights, weights):\n', Cov(heights, weights))
    print('Corr(heights, weights):\n', Corr(heights, weights))
    print('SpearmanCorr(heights, weights):\n', SpearmanCorr(heights, weights))
    print('Corr(heights, np.log(weights)):\n', Corr(heights, np.log(weights)))

    ## make nsfg DataFrame
    import first

    live, firsts, others = first.MakeFrames()
    live = live.dropna(subset=['agepreg', 'totalwgt_lb'])

    ## make scatter plot of birth weight vs. mother's age
    thinkplot.CorrelationPlots(live, 'agepreg', 'totalwgt_lb')
    thinkplot.Show()
