## Module import and boilerplate
import first
import hinc
import hinc2
import math
import brfss
import thinkstats2
import thinkplot
import numpy as np

## function and class definition
def RawMoment(xs, k):
    return sum(x**k for x in xs) / len(xs)

def CentralMoment(xs, k):
    mean = RawMoment(xs, 1)
    return sum((x-mean)**k for x in xs) / len(xs)

def StandardizedMoment(xs, k):
    var = CentralMoment(xs, 2)
    std = math.sqrt(var)
    return CentralMoment(xs, k) / std**k

def Skewness(xs):
    return StandardizedMoment(xs, 3)

def Mean(xs):
    return RawMoment(xs, 1)

def StDev(xs):
    var = CentralMoment(xs, 2)
    return math.sqrt(var)

def Median(xs):
    cdf = thinkstats2.Cdf(xs)
    return cdf.Value(0.5)

def PearsonMedianSkewness(xs):
    median = thinkstats2.Median(xs)
    mean = RawMoment(xs, 1)
    var = CentralMoment(xs, 2)
    std = math.sqrt(var)
    gp = 3 * (mean - median) / std
    return gp

def Kurtosis(xs):
    return StandardizedMoment(xs, 4)

def SampleExcessKertosis(xs):
    return StandardizedMoment(xs, 4) - 3

def SampleStatistics(xs):
    print('Mean:\n',Mean(xs))
    print('Median:\n',Median(xs))
    print('Standard Deviation:\n', StDev(xs))
    print('Skewness:\n',Skewness(xs))
    print('Pearson Median Skewness:\n',PearsonMedianSkewness(xs))
    print('Kurtosis:\n',Kurtosis(xs))

if __name__ == '__main__':
    """Main scripts here"""
    ## make brfss dataframes
    df = brfss.ReadBrfss(nrows=None)
    female = df[df.sex == 2]
    female_heights = female.htm3.dropna()

    ## female height statistics
    mean, std = female_heights.mean(), female_heights.std()
    print('mean:\n', mean)
    print('std:\n', std)

    ## make pdf representing female distribution
    pdf = thinkstats2.NormalPdf(mean, std)
    pmf = pdf.MakePmf()
    thinkplot.PrePlot(2)
    thinkplot.Pdf(pdf, label='normal pdf')

    thinkplot.Pmf(pmf, label='normal pmf')
    thinkplot.Show(xlabel='x', xlim=[140, 186])

    ## KDE of normal pdf
    i = 6
    thinkplot.PrePlot(i+1)
    thinkplot.Pdf(pdf, label='normal')

    for _ in range(i):
        sample = np.random.normal(mean, std, 500)
        sample_pdf = thinkstats2.EstimatedPdf(sample, label='sample')
        thinkplot.Pdf(sample_pdf, label='sample KDE')

    thinkplot.Show(xlabel='x', ylabel='PDF', xlim=[140, 186])

    ## calculate moments
    print('RawMoment')
    print(RawMoment(female_heights, 1), RawMoment(female_heights, 2))
    print('\n CentralMoment')
    print(CentralMoment(female_heights, 1), CentralMoment(female_heights, 2), CentralMoment(female_heights, 3))
    print('\n StandardizedMoment')
    print(StandardizedMoment(female_heights, 1), StandardizedMoment(female_heights, 2), StandardizedMoment(female_heights, 3))

    ## make frames of birth data
    live, firsts, others  = first.MakeFrames()
    birth_weights = live.totalwgt_lb.dropna()

    ## make pdf of birth weight and calculate statistics
    pdf = thinkstats2.EstimatedPdf(birth_weights)
    thinkplot.Pdf(pdf, label='birth weight')
    thinkplot.Show(xlabel='PDF', ylabel='lbs')

    ## make adult weight data frames
    adult_weights = df.wtkg2.dropna()

    ## evaluate skewness of adult weights
    pdf = thinkstats2.EstimatedPdf(adult_weights)
    thinkplot.Pdf(pdf, label='Adult weight')
    thinkplot.Show(xlabel='Adult weight (kg)', ylabel='PDF')

    ## weight kurtosis
    print('Kurtosis(adult_weights):\n', Kurtosis(adult_weights))
    print('SampleExcessKertosis(adult_weights):\n', SampleExcessKertosis(adult_weights))

    ## compute statistics of income data
    df = hinc.ReadData()
    log_sample = hinc2.InterpolateSample(df, log_upper=6.0)

    ## Convert sample from log $ to $
    sample = np.power(10, log_sample)
    cdf = thinkstats2.Cdf(sample, label='interp. data')
    thinkplot.Cdf(cdf)
    thinkplot.Show(xlabel='Income ($)',
                   ylabel='CDF')

    ## Compute statistics
    SampleStatistics(sample)

