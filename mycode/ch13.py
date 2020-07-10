## module import and boilerplate

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import random

import thinkstats2
import thinkplot

import nsfg
import survival

## function and class definition

def MakeSurvivalFromCdf(cdf, label=''):
    """Make survival function based on CDF.

    cdf: Cdf

    return: SurvivalFunction
    """
    ts = cdf.xs
    ss = 1 - cdf.ps
    return survival.SurvivalFunction(ts, ss, label)

def EstimateMarriageSurvival(resp):
    """Estimates the survival curve.

    resp: DataFrame of respondents

    return: pair of HazardFunction, SurvivalFunction
    """
    # NOTE: filling in missing values would be better than dropping them
    complete = resp6[resp6.evrmarry==1].agemarry.dropna()
    ongoing = resp6[resp6.evrmarry==0].age

    hf = survival.EstimateHazardFunction(complete, ongoing)
    sf = hf.MakeSurvival()

    return hf, sf

def EstimateDivorceSurvival(resp):
    """Estimates the survival curve of marriages that end in divorce.

    resp: DataFrame of respondents

    return: pair of HazardFunction, SurvivalFunction
    """
    # NOTE: filling in missing values would be better than dropping them
    complete = resp[resp.notdivorced==0].duration.dropna()
    ongoing = resp[resp.notdivorced==1].durationsofar.dropna()

    hf = survival.EstimateHazardFunction(complete, ongoing)
    sf = hf.MakeSurvival()

    return hf, sf

def ResampleSurvival(resp, iters=101):
    """Resamples respondents and estimates survival function.

    resp: DataFrame of respondents
    iters: number of resamples
    """
    _, original_sf = EstimateMarriageSurvival(resp)

    low, high = resp.agemarry.min(), resp.agemarry.max()
    ts = np.arange(low, high, 1/12.0)

    ss_seq  = []
    for _ in range(iters):
        sample = thinkstats2.ResampleRowsWeighted(resp)
        _, sf = EstimateMarriageSurvival(sample)
        ss_seq.append(sf.Probs(ts))

    low, high = thinkstats2.PercentileRows(ss_seq, [5, 95])
    thinkplot.Plot(original_sf)
    thinkplot.FillBetween(ts, low, high, color='gray', label='90% CI')
    thinkplot.Show(xlabel='Age (years)',
                   ylabel='Prob unmarried',
                   xlim=[12, 46],
                   ylim=[0,1])

def AddLabelsByDecade(groups, **options):
    """Draws fake points in order to add labels to the legend

    groups: GroupBy object
    """
    thinkplot.PrePlot(len(groups))
    for name, _ in groups:
        label = '%d0s' % name
        thinkplot.Plot([15], [1], label=label, **options)

def EstimateDivorceSurvivalByDecade(groups, **options):
    """Groups respondents by decade and plots survival curves.

    groups: GroupBy object
    """
    thinkplot.PrePlot(len(groups))
    for _, group in groups:
        _, sf = EstimateDivorceSurvival(group)
        thinkplot.Plot(sf, **options)

def EstimateMarriageSurvivalByDecade(groups, **options):
    """Groups respondents by decade and plots survival curves.

    groups: GroupBy object
    """
    thinkplot.PrePlot(len(groups))
    for _, group in groups:
        _, sf = EstimateMarriageSurvival(group)
        thinkplot.Plot(sf, **options)

def PlotDivorcePredictionsByDecade(groups, **options):
    """Groups respondents by decade and plots survival curves.

    groups: GroupBy object
    """
    hfs = []
    for _, group in groups:
        hf, sf = EstimateDivorceSurvival(group)
        hfs.append(hf)

    thinkplot.PrePlot(len(hfs))
    for i, hf in enumerate(hfs):
        if i > 0:
            hf.Extend(hfs[i-1])
        sf = hf.MakeSurvival()
        thinkplot.Plot(sf, **options)

def PlotPredictionsByDecade(groups, **options):
    """Groups respondents by decade and plots survival curves.

    groups: GroupBy object
    """
    hfs = []
    for _, group in groups:
        hf, sf = EstimateMarriageSurvival(group)
        hfs.append(hf)

    thinkplot.PrePlot(len(hfs))
    for i, hf in enumerate(hfs):
        if i > 0:
            hf.Extend(hfs[i-1])
        sf = hf.MakeSurvival()
        thinkplot.Plot(sf, **options)

def PlotDivorceResampledByDecade(resps, iters=11, group_str='decade', predict_flag=False, omit=None):
    """Plots survival curves for resampled data.

    resps: list of DataFrames
    iters: number of resamples to plot
    predict_flag: whether to also plot predictions
    """
    for i in range(iters):
        samples = [thinkstats2.ResampleRowsWeighted(resp)
                  for resp in resps]
        sample = pd.concat(samples, ignore_index=True)
        groups = sample.groupby(group_str)

        if omit:
            groups = [(name, group) for name, group in groups
                      if name not in omit]

        if i == 0:
            AddLabelsByDecade(groups, alpha=0.7)

        if predict_flag:
            PlotDivorcePredictionsByDecade(groups, alpha=0.1)
            EstimateDivorceSurvivalByDecade(groups, alpha=0.1)
        else:
            EstimateDivorceSurvivalByDecade(groups, alpha=0.2)

def PlotResampledByDecade(resps, iters=11, group_str='decade', predict_flag=False, omit=None):
    """Plots survival curves for resampled data.

    resps: list of DataFrames
    iters: number of resamples to plot
    predict_flag: whether to also plot predictions
    """
    for i in range(iters):
        samples = [thinkstats2.ResampleRowsWeighted(resp)
                  for resp in resps]
        sample = pd.concat(samples, ignore_index=True)
        groups = sample.groupby(group_str)

        if omit:
            groups = [(name, group) for name, group in groups
                      if name not in omit]

        if i == 0:
            AddLabelsByDecade(groups, alpha=0.7)

        if predict_flag:
            PlotPredictionsByDecade(groups, alpha=0.1)
            EstimateMarriageSurvivalByDecade(groups, alpha=0.1)
        else:
            EstimateMarriageSurvivalByDecade(groups, alpha=0.2)

def CleanData(resp):
    """Cleans respondent data for divorce analysis

    resp: DataFrame
    """
    resp.cmdivorcx.replace([9998, 9999], np.nan, inplace=True)

    resp['notdivorced'] = resp.cmdivorcx.isnull().astype(int)
    resp['duration'] = (resp.cmdivorcx - resp.cmmarrhx) / 12.0
    resp['durationsofar'] = (resp.cmintvw - resp.cmmarrhx) / 12.0

    month0 = pd.to_datetime('1899-12-15')
    dates = [month0 + pd.DateOffset(months=cm) for cm in resp.cmbirth]
    resp['decade'] = (pd.DatetimeIndex(dates).year - 1900) // 10
    resp['agefirstmarr'] = (resp6.cmmarrhx - resp6.cmbirth) / 12.0 // 10

## main scripts
if __name__ == '__main__':
    """main scripts"""

    ## make survival plot of pregnancy lengths
    preg = nsfg.ReadFemPreg()
    complete = preg.query('outcome in [1,3,4]').prglngth
    cdf = thinkstats2.Cdf(complete, label='cdf')

    sf = MakeSurvivalFromCdf(cdf, label='Survival')

    thinkplot.Plot(sf)
    thinkplot.Cdf(cdf, alpha=0.2)
    thinkplot.Show(loc='center left')

    ## calculate hazard function
    hf = sf.MakeHazardFunction(label='hazard')
    thinkplot.Plot(hf)
    thinkplot.Show(ylim=[0, 0.75], loc='upper left')

    #########################################
    ## Age at first marriage
    #########################################

    # clean dataframe and extract sub-groups we need
    resp6 = nsfg.ReadFemResp()
    resp6.cmmarrhx.replace([9997, 9998, 9999], np.nan, inplace=True)
    resp6['agemarry'] = (resp6.cmmarrhx - resp6.cmbirth) / 12.0
    resp6['age'] = (resp6.cmintvw - resp6.cmbirth) / 12.0

    complete = resp6[resp6.evrmarry==1].agemarry.dropna()
    ongoing = resp6[resp6.evrmarry==0].age

    ## estimate hazard function
    hf = survival.EstimateHazardFunction(complete, ongoing)
    thinkplot.Plot(hf)
    thinkplot.Show(xlabel='Age (years)',
                   ylabel='Hazard')

    ## make survival function from hazard function
    sf = hf.MakeSurvival()
    thinkplot.Plot(sf)
    thinkplot.Show(xlabel='Age (years)',
                   ylabel='Prob unmarried',
                   ylim=[0,1])

    ## compute resampling survival with weights
    ResampleSurvival(resp6)

    ## show more data
    resp5 = survival.ReadFemResp1995()
    resp6 = survival.ReadFemResp2002()
    resp7 = survival.ReadFemResp2010()

    resps = [resp5, resp6, resp7]

    ## plot survival by decade
    PlotResampledByDecade(resps)
    thinkplot.Show(xlabel='Age (years)',
                   ylabel='Prob unmarried',
                   xlim=[12, 46],
                   ylim=[0,1])

    ## calculate expected remaining lifetime for pregnancies
    preg = nsfg.ReadFemPreg()

    complete = preg.query('outcome in [1,3,4]').prglngth
    print('Number of complete pregnancies', len(complete))
    ongoing = preg[preg.outcome==6].prglngth
    print('Number of ongoing pregnancies', len(ongoing))

    hf = survival.EstimateHazardFunction(complete, ongoing)
    sf1 = hf.MakeSurvival()

    ## plot remaininig lifetime
    rem_life1 = sf1.RemainingLifetime()
    thinkplot.Plot(rem_life1)
    thinkplot.Show(title='Remaining pregnancy length',
                   xlabel='Weeks',
                   ylabel='Mean remaining weeks')

    ## calculate expected remaining single time until marriage
    hf, sf2 = EstimateMarriageSurvival(resp6)
    func = lambda pmf: pmf.Percentile(50)
    rem_life2 = sf2.RemainingLifetime(filler=np.inf, func=func)

    thinkplot.Plot(rem_life2)
    thinkplot.Show(title='Years until first marriage',
                   ylim=[0, 15],
                   xlim=[11, 31],
                   xlabel='Age (years)',
                   ylabel='Median remaining years')

    #####################################################
    # Exercises
    #####################################################
    CleanData(resp6)
    married6 = resp6[resp6.evrmarry==1]

    CleanData(resp7)
    married7 = resp7[resp7.evrmarry==1]

    marrieds = [married6, married7]

    ## make survival function of marriage duration
    hf, sf = EstimateDivorceSurvival(married6)
    thinkplot.Plot(sf)
    thinkplot.Show(xlabel='Marriage duration (years)',
                   ylabel='Probability marriage has ended',
                   ylim=[0,1])

    ## estimate remaining marriage life
    filler = sf.ts.max()
    func = lambda pmf: pmf.Percentile(25)
    rem_life = sf.RemainingLifetime(filler=filler, func=func)

    thinkplot.Plot(rem_life)
    thinkplot.Show(title='Years remaining in marriage',
                   #ylim=[0, 30],
                   #xlim=[0, 30],
                   xlabel='Years married',
                   ylabel='Median remaining years')

    ## plot survival by decade
    PlotDivorceResampledByDecade(marrieds, predict_flag=False, group_str='agefirstmarr', omit=[3])
    thinkplot.Show(xlabel='Length of marriage',
                   ylabel='Probability marriage ended',
                   xlim=[0, 32],
                   ylim=[0,1])
