## module import and boilerplate
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import random

import thinkstats2
import thinkplot

## function and class definition

def GroupByDay(transactions, func=np.mean):
    """Group transactions by day and compute daily mean ppg.

    transactions: DataFrame of transactions

    return: DataFrame of daily prices
    """
    grouped = transactions[['date','ppg']].groupby('date')
    daily = grouped.aggregate(func)

    daily['date'] = daily.index
    start = daily.date[0]
    one_year = np.timedelta64(1, 'Y')
    daily['years'] = (daily.date - start) / one_year

    return daily

def GroupByQualityAndDay(transactions):
    """Divides transactions by quality and computes mean daily price.

    transactions: DataFrame of transactions

    return: map from quality to time series of ppg
    """
    groups = transactions.groupby('quality')
    dailies = {}
    for name, group in groups:
        dailies[name] = GroupByDay(group)

    return dailies

def RunLinearModel(daily):
    model = smf.ols('ppg ~ years', data=daily)
    results = model.fit()
    return model, results

def RunQuadraticModel(daily):
    daily['years2'] = daily.years ** 2
    model = smf.ols('ppg ~ years + years2', data=daily)
    results = model.fit()
    return model, results

def PlotFittedValues(model, results, label=''):
    """Plots original data and fitted values

    model: StatsModel model object
    results: StatsModel results object
    """
    years = model.exog[:,1]
    values = model.endog
    thinkplot.Scatter(years, values, s=15, label=label)
    thinkplot.Plot(years, results.fittedvalues, label='model', color='#ff7f00')

def PlotLinearModel(daily, name):
    """Plots linear fit to a sequence of prices, and the residuals.

    daily: DataFrame of daily prices
    name: string
    """
    model, results = RunLinearModel(daily)
    PlotFittedValues(model, results, label=name)
    thinkplot.Config(title='Fitted values',
                     xlabel='Years',
                     xlim=[-0.1, 3.8],
                     ylabel='Price per gram ($)')
    thinkplot.Show()

def PlotRollingMean(daily, name):
    """Plots rolling mean.

    daily: DataFrame of daily prices
    name: string
    """
    dates = pd.date_range(daily.index.min(), daily.index.max())
    reindexed = daily.reindex(dates)

    thinkplot.Scatter(reindexed.ppg, s=15, alpha=0.2, label=name)
    roll_mean = reindexed.ppg.rolling(30).mean()
    thinkplot.Plot(roll_mean, label='rolling mean', color='#ff7f00')
    plt.xticks(rotation=30)
    thinkplot.Config(ylabel='price per gram ($)')
    thinkplot.Show()

def PlotEWMA(daily, name):
    """Plot rolling mean.

    daily: DataFrame of daily prices
    name: string
    """
    dates = pd.date_range(daily.index.min(), daily.index.max())
    reindexed = daily.reindex(dates)

    thinkplot.Scatter(reindexed.ppg, s=15, alpha=0.2, label=name)
    roll_mean = reindexed.ppg.ewm(30).mean()
    thinkplot.Plot(roll_mean, label='EWMA', color='#ff7f00')
    plt.xticks(rotation=30)
    thinkplot.Config(ylabel='price per gram ($)')
    thinkplot.Show()

def FillMissing(daily, span=30):
    """Fills missing values with an exponentially weighted moving average.

    Resulting DataFrame has new columns 'ewma' and 'resid'.

    daily: DataFrame of daily prices
    span: window size (sort of) passed to ewma

    return: new DataFrame of daily prices
    """
    dates = pd.date_range(daily.index.min(), daily.index.max())
    reindexed = daily.reindex(dates)

    ewma = reindexed.ppg.ewm(span=span).mean()

    resid = (reindexed.ppg - ewma).dropna()
    fake_data = ewma + thinkstats2.Resample(resid, len(reindexed))
    reindexed.ppg.fillna(fake_data, inplace=True)

    reindexed['ewma'] = ewma
    reindexed['resid'] = reindexed.ppg - ewma
    return reindexed

def PlotFilled(daily, name):
    """Plot the EWMA and filled data.

    daily: DataFrame of daily prices
    name: string
    """
    filled = FillMissing(daily, span=30)
    thinkplot.Scatter(filled.ppg, s=15, alpha=0.2, label=name)
    thinkplot.Plot(filled.ewma, label='EWMA', color='#ff7f00')
    plt.xticks(rotation=30)
    thinkplot.Config(label='Price per gram ($)')
    thinkplot.Show()

def SerialCorr(series, lag=1):
    xs = series[lag:]
    ys = series.shift(lag)[lag:]
    corr = thinkstats2.Corr(xs, ys)
    return corr

def SimulateAutocorrelation(daily, iters=1001, nlags=40):
    """Resample residuals, compute autocorrelation, and plot percentiles

    daily: DataFrame
    iters: number of simulations to run
    nlags: maximum lags to compute autocorrelation
    """
    t = []
    for _ in range(iters):
        filled = FillMissing(daily, span=30)
        resid = thinkstats2.Resample(filled.resid)
        acf = smtsa.acf(resid, nlags=nlags, unbiased=True)[1:]
        t.append(np.abs(acf))

    high = thinkstats2.PercentileRows(t, [97.5])[0]
    low = -high
    lags = range(1, nlags+1)
    thinkplot.FillBetween(lags, low, high, alpha=0.2, color='gray')

def PlotAutoCorrelation(dailies, nlags=40, add_weekly=False):
    """Plots autocorrelation functions.

    dailies: map from category name to DataFrame of daily prices
    nlags: number of lags to compute
    add_weekly: boolean, whether to add simulated weekly pattern
    """
    thinkplot.PrePlot(3)
    daily = dailies['high']
    SimulateAutocorrelation(daily)

    for name, daily in dailies.items():

        if add_weekly:
            daily = AddWeeklySeasonality(daily)

        filled = FillMissing(daily, span=30)

        acf = smtsa.acf(filled.resid, nlags=nlags, unbiased=True)
        lags = np.arange(len(acf))
        thinkplot.Plot(lags[1:], acf[1:], label=name)

def AddWeeklySeasonality(daily):
    """Adds a weekly pattern.

    daily: DataFrame of daily prices

    returns: new DataFrame of daily prices
    """
    fri_or_sat = (daily.index.dayofweek==4) | (daily.index.dayofweek==5)
    fake = daily.copy()
    fake.loc[fri_or_sat, 'ppg'] += np.random.uniform(0, 2, fri_or_sat.sum())
    return fake

def GenerateSimplePrediction(results, years):
    """Generate a simple prediction.

    result: results object
    years: sequence of times (in years) to make predictions for

    return: sequence of predicted values
    """
    n = len(years)
    inter = np.ones(n)
    d = dict(Intercept=inter, years=years, years2=years**2)
    predict_df = pd.DataFrame(d)
    predict = results.predict(predict_df)
    return predict

def PlotSimplePrediction(results, years):
    predict = GenerateSimplePrediction(results, years)

    thinkplot.Scatter(daily.years, daily.ppg, alpha=0.2, label=name)
    thinkplot.Plot(years, predict, color='#ff7f00')
    xlim = years[0] - 0.1, years[-1] + 0.1
    thinkplot.Show(title='Predictions',
                   xlabel='Years',
                   xlim=xlim,
                   ylabel='Price per gram ($)',
                   loc='upper right')

def SimulateResults(daily, iters=101, func=RunLinearModel):
    """Run simulations based on resampling residuals.

    daily: DataFrame of daily prices
    iters: number of simulations
    func: function that fits a model to the data

    return: list of result objects
    """
    _, results = func(daily)
    fake = daily.copy()

    result_seq = []
    for _ in range(iters):
        fake.ppg = results.fittedvalues + thinkstats2.Resample(results.resid)
        _, fake_results = func(fake)
        result_seq.append(fake_results)

    return result_seq

def GeneratePredictions(result_seq, years, add_resid=False):
    """Generates an array of predicted values from a list of model results.

    When add_resid is False, predictions represent sampling error only.

    When add_resid is True, they also include residual error
    (which is more relevant to prediction).

    result_seq: list of model results
    years: sequence of times (in years) to make predictions for
    add_resid: boolean, whether to add resampled residuals

    return: sequence of predictions
    """
    n = len(years)
    d = dict(Intercept=np.ones(n), years=years, years2=years**2)
    predict_df = pd.DataFrame(d)

    predict_seq = []
    for fake_results in result_seq:
        predict = fake_results.predict(predict_df)
        if add_resid:
            predict += thinkstats2.Resample(fake_results.resid, n)
        predict_seq.append(predict)

    return predict_seq

def PlotPredictions(daily, years, iters=101, percent=90, func=RunLinearModel):
    """Plot predictions.

    daily: DataFrame of daily prices
    years: sequence of times (in years) to make predictions for
    iters: number of simulations
    percent: what percentile range to show
    func: function that fits a model to data
    """
    result_seq = SimulateResults(daily, iters=iters, func=func)
    p = (100 - percent) / 2
    percents = p, 100-p

    predict_seq = GeneratePredictions(result_seq, years, add_resid=True)
    low, high = thinkstats2.PercentileRows(predict_seq, percents)
    thinkplot.FillBetween(years, low, high, alpha=0.3, color='gray')

    predict_seq = GeneratePredictions(result_seq, years, add_resid=False)
    low, high = thinkstats2.PercentileRows(predict_seq, percents)
    thinkplot.FillBetween(years, low, high, alpha=0.5, color='gray')

def SimulateIntervals(daily, iters=101, func=RunLinearModel):
    """Run simulations based on different subsets of the data.

    daily: DataFrame of daily prices
    iters: number of simulations
    func: function that fits a model to the data

    return: list of result objects.
    """
    result_seq = []
    starts = np.linspace(0, len(daily), iters, dtype=int)

    for start in starts[:-2]:
        subset = daily[start:]
        _, results = func(subset)
        fake = subset.copy()

        for _ in range(iters):
            fake.ppg = (results.fittedvalues +
                        thinkstats2.Resample(results.resid))
            _, fake_results = func(fake)
            result_seq.append(fake_results)

    return result_seq

def PlotIntervals(daily, years, iters=101, percent=90, func=RunLinearModel):
    """Plots predictions based on different intervals.

    daily: DataFrame of daily prices
    years: sequence of times (in years) to make predictions for
    iters: number of simulations
    percent: what percentile range to show
    func: function that fits a model to the data
    """
    result_seq = SimulateIntervals(daily, iters=iters, func=func)
    p = (100 - percent) / 2
    percents = p, 100-p

    predict_seq = GeneratePredictions(result_seq, years, add_resid=True)
    low, high = thinkstats2.PercentileRows(predict_seq, percents)
    thinkplot.FillBetween(years, low, high, alpha=0.2, color='gray')

class SerialCorrelationTest(thinkstats2.HypothesisTest):

    def __init__(self, data, lag=1):
        self.data = data
        self.lag = lag
        self.MakeModel()
        self.actual = self.TestStatistic(data)

    def TestStatistic(self, data):
        test_stat =  SerialCorr(data, lag=self.lag)
        return test_stat

    def MakeModel(self):
        self.fake_data = self.data.copy()

    def RunModel(self):
        np.random.shuffle(self.fake_data)
        return self.fake_data

## main scripts
if __name__ == '__main__':

    ## read csv and group by quality and day
    transactions = pd.read_csv('mj-clean.csv', parse_dates=[5])
    dailies = GroupByQualityAndDay(transactions)

    ## plot time series by quality
    thinkplot.PrePlot(rows=3)
    for i, (name, daily) in enumerate(dailies.items()):
        thinkplot.SubPlot(i+1)
        title = 'Price per gram ($)' if i==0 else ''
        thinkplot.Config(ylim=[0,20], title=title)
        thinkplot.Scatter(daily.ppg, s=10, label=name)
        if i==2:
            plt.xticks(rotation=30)
            thinkplot.Config()
        else:
            thinkplot.Config(xticks=[])

    plt.show()

    ## calculate linear regressions for each quality

    for name, daily in dailies.items():
        model, results = RunLinearModel(daily)
        print(name)
        print(results.summary())

    ## plot fitted values for 'high'
    name='high'
    daily = dailies[name]
    PlotLinearModel(daily, name)

    ## plot rolling mean
    PlotRollingMean(daily, name)

    ## plot EWMA
    PlotEWMA(daily, name)

    ## plot filled
    PlotFilled(daily, name)

    ## fill missing values
    filled_dailies = {}
    for name, daily in dailies.items():
        filled_dailies[name] = FillMissing(daily, span=30)

    ## calculate serial correlation
    for name, filled in filled_dailies.items():
        corr = SerialCorr(filled.ppg, lag=1)
        print(name, corr)

    ## calculate serial correlation after subtracting away trends
    for name, filled in filled_dailies.items():
        corr = SerialCorr(filled.resid, lag=1)
        print(name, corr)

    ## calculate serial correlation across common time periods
    rows = []
    for lag in [1, 7, 30, 365]:
        print(lag, end='\t')
        for name, filled in filled_dailies.items():
            corr = SerialCorr(filled.resid, lag)
            print('%.2g' %corr, end='\t')
        print()

    ## use autocorrelation
    import statsmodels.tsa.stattools as smtsa

    filled = filled_dailies['high']
    acf = smtsa.acf(filled.resid, nlags=365, unbiased=True)
    print('%0.2g, %0.2g, %0.2g, %0.2g, %0.2g' %
            (acf[0], acf[1], acf[7], acf[30], acf[365]))

    ## make acf plot
    axis = [0, 41, -0.2, 0.2]

    PlotAutoCorrelation(dailies, add_weekly=False)
    thinkplot.Show(axis=axis,
                   loc='lower right',
                   ylabel='correlation',
                   xlabel='lag (day)')

    ## make acf plot with simulated weekly trends
    axis = [0, 41, -0.2, 0.2]

    PlotAutoCorrelation(dailies, add_weekly=True)
    thinkplot.Show(axis=axis,
                   loc='lower right',
                   ylabel='correlation',
                   xlabel='lag (day)')

    ## make prediction for high-quality category
    name = 'high'
    daily = dailies[name]

    _, results = RunLinearModel(daily)
    years = np.linspace(0, 5, 101)
    PlotSimplePrediction(results, years)

    ## plot predictions for high quality
    years = np.linspace(0, 5, 101)
    thinkplot.Scatter(daily.years, daily.ppg, alpha=0.1, label=name)
    PlotPredictions(daily, years)
    xlim = years[0]-0.1, years[-1]+0.1
    thinkplot.Show(title='Predictions',
                   xlabel='Years',
                   xlim=xlim,
                   ylabel='Price per gram ($)')

    ## make predictions of high quality with different start points
    name = 'high'
    daily = dailies[name]

    thinkplot.Scatter(daily.years, daily.ppg, alpha=0.1, label=name)
    PlotIntervals(daily, years)
    PlotPredictions(daily, years)
    xlim = years[0]-0.1, years[-1]+0.1
    thinkplot.Show(title='Predictions',
                   xlabel='Years',
                   xlim=xlim,
                   ylabel='Price per gram ($)')

    ## make predictions with quadratic model
    years = np.linspace(0, 5, 101)
    thinkplot.Scatter(daily.years, daily.ppg, alpha=0.1, label=name)
    PlotPredictions(daily, years, func=RunQuadraticModel)
    xlim = years[0]-0.1, years[-1]+0.1
    thinkplot.Show(title='Predictions',
                   xlabel='Years',
                   xlim=xlim,
                   ylabel='Price per gram ($)')

    ## serial correlation statistical significance
    name = 'high'
    daily = dailies[name]
    data = daily.ppg

    ht = SerialCorrelationTest(data)
    pvalue = ht.PValue(iters=100)
    print('pvalue:\n',pvalue)
    ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF')

    ## now test with residuals
    filled_dailies = {}
    for name, daily in dailies.items():
        filled_dailies[name] = FillMissing(daily, span=30)
    name = 'high'
    daily = filled_dailies[name]
    data = daily.resid

    ht = SerialCorrelationTest(data)
    pvalue = ht.PValue(iters=100)
    print('pvalue:\n',pvalue)
    ht.PlotCdf()
    thinkplot.Show(xlabel='test statistic',
                   ylabel='CDF')
