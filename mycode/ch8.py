## module import and boilerplate
import numpy as np
import brfss

import random
import thinkplot
import thinkstats2

## function and class definition
def RMSE(estimates, actual):
    """Computes the root mean squared error of a sequence of estimates.

    estimate: sequence of numbers
    actual: float actual value

    returns: float RMSE
    """
    e2 = [(estimate-actual)**2 for estimate in estimates]
    mse = np.mean(e2)
    return np.sqrt(mse)

def MeanError(estimates, actual):
    """Computes the mean error of a sequence of estimates.

    estimate: sequence of numbers
    actual: float actual value

    returns: float mean error
    """
    errors = [(estimate-actual) for estimate in estimates]
    return np.mean(errors)

def Estimate1(n=7, iters=1000):
    """Evaluates the RMSE of sample mean and median as estimators.

    n: sample size
    iters: number of iterations
    """
    mu = 0
    sigma = 1

    means = []
    medians = []

    for _ in range(iters):
        xs = [random.gauss(mu, sigma) for _ in range(n)]
        xbar = np.mean(xs)
        median = np.median(xs)
        means.append(xbar)
        medians.append(median)

    print('Experiment 1')
    print('MeanError(means, mu):\n',MeanError(means, mu))
    print('MeanError(medians, mu):\n',MeanError(medians, mu))
    print('RMSE xbar', RMSE(means, mu))
    print('RMSE median', RMSE(medians, mu))

def Estimate2(n=7, iters=1000):
    """Evaluates the RMSE of sample mean and median as estimators.

    n: sample size
    iters: number of iterations
    """
    mu = 0
    sigma = 1

    estimates1 = []
    estimates2 = []

    for _ in range(iters):
        xs = [random.gauss(mu, sigma) for _ in range(n)]
        biased = np.var(xs)
        unbiased = np.var(xs, ddof=1)
        estimates1.append(biased)
        estimates2.append(unbiased)

    print('Experiment 2')
    print('mean error biased', MeanError(estimates1, sigma**2))
    print('mean error unbiased', MeanError(estimates2, sigma**2))
    print('RMSE variance - biased', RMSE(estimates1, sigma**2))
    print('RMSE variance - unbiased', RMSE(estimates2, sigma**2))

def SimulateSampleNormal(mu=90, sigma=7.5, n=9, iters=1000):
    """Simulate samples of normal dist of 'mu' and 'sigma'
    of size 'n' for 'm' iters.
    mu: measured sample mean
    sigma: measured sample st. dev.
    n: sample size
    iters: number of iterations

    return: xbars - list of sample means
    """
    xbars = []
    for j in range(iters):
        xs = np.random.normal(mu, sigma, n)
        xbar = np.mean(xs)
        xbars.append(xbar)

    return xbars

def SimulateSampleExpo(lam=2.0, n=10, iters=1000):
    """Simulate samples of exponential dist of lambda 'lam'
    of size 'n' for 'm' iters.
    lam: float shape parameter
    n: sample size
    iters: number of iterations

    return:
         Ls - estimates of lam based on mean
         Lms - estimates of lam based on median
    """
    Ls = []
    Lms = []
    for j in range(iters):
        xs = np.random.exponential(1.0/lam, n)
        L = 1/ np.mean(xs)
        Lm = np.log(2) / thinkstats2.Median(xs)
        Ls.append(L)
        Lms.append(Lm)

    return Ls, Lms

def Estimate3(n=7, iters=1000):
    """Evaulates sample mean and sample median as estimators for properties of
    exponential distribution.
    n: int sample size
    iters: int number of iterations

    return: None
    """
    lam = 2

    means = []
    medians = []
    for _ in range(iters):
        xs = np.random.exponential(1.0/lam, n)
        L = 1/np.mean(xs)
        Lm = np.log(2) / thinkstats2.Median(xs)
        means.append(L)
        medians.append(Lm)

    print('RMSE(means, lam):\n',RMSE(means, lam))
    print('RMSE(medians, lam):\n',RMSE(medians, lam))
    print('MeanError(means, lam):\n',MeanError(means, lam))
    print('MeanError(medians, lam):\n',MeanError(medians, lam))

def SimulateGame(lam):
    """Simulates hockey game with time between goals for one team
    distributed along exponential distribution of shape 'lam'.
    Game ends when time exceeds 1.
    lam: float shape parameter

    returns: int number of goals
    """
    t = 0
    goals = 0
    while True:
        dt = np.random.exponential(1.0/lam)
        t += dt
        if t > 1:
            break
        goals += 1

    L = goals
    return L

def EstimateLamFromGames(lam=4, iters=1000):
    """Uses hockey simulation to estimate exponential scale parameter 'lam'.

    lam: float goals per game
    iters: number of iterations
    """
    Ls = []
    for _ in range(iters):
        L = SimulateGame(lam)
        Ls.append(L)

    print('np.mean(Ls):\n',np.mean(Ls))
    print('RMSE(Ls, lam):\n',RMSE(Ls, lam))
    print('MeanError(Ls, lam):\n',MeanError(Ls, lam))

def HockeyCompetition(lam1=4, lam2=3, games=82):
    """Simulates hockey games between two teams with exponential goal scoring
    distributions of 'lam1' and 'lam2'.
    lam1: team 1 scale parameter
    lam2: team 2 scale parameter
    games: number of games between teams
    """
    wins1 = 0
    wins2 = 0
    ties = 0
    for _ in range(games):
        goals1 = SimulateGame(lam1)
        goals2 = SimulateGame(lam2)

        if goals1 > goals2:
            wins1 += 1
        elif goals2 > goals1:
            wins2 += 1
        elif goals1 == goals2:
            ties += 1

    print('Team 1 Record:')
    print('%d wins \n%d losses \n%d ties' %(wins1, wins2, ties))

## main scripts
if __name__ == '__main__':
    """main scripts"""

    ## Experiment 1
    Estimate1()

    ## Experiment 2
    Estimate2()

    ## SimulateSampleNormal
    ns = np.arange(5,20, step=3)
    thinkplot.PrePlot(len(ns))
    conf = 90

    for n in ns:
        xbars = SimulateSampleNormal(n=n)

        cdf = thinkstats2.Cdf(xbars)
        low = (100-conf)/2
        ci = cdf.Percentile(low), cdf.Percentile(100-low)
        label='n=%d' %n
        thinkplot.Cdf(cdf, label=label)
        thinkplot.Vlines(ci, -1,2, alpha=0.8, color='0.5')

        print('\n Sample Size: %d' %n)
        print('np.mean(xbars):\n',np.mean(xbars))
        print('ci:\n',ci)
        stderr = RMSE(xbars, 90)
        print('stderr:\n',stderr)

    thinkplot.Config(xlabel='Sample mean',
                     ylabel='CDF',
                     ylim=[-0.05, 1.05])
    thinkplot.Show()

    ## Estimate3
    Estimate3()

    ## SimulateSampleExpo
    ns = np.arange(5,20, step=3)
    conf = 90
    lam=2

    thinkplot.PrePlot(len(ns))
    stderrs = []

    for n in ns:
        Ls, Lms = SimulateSampleExpo(lam=lam,n=n)

        cdf = thinkstats2.Cdf(Ls)
        low = (100-conf)/2
        ci = cdf.Percentile(low), cdf.Percentile(100-low)
        label='n=%d' %n
        thinkplot.Cdf(cdf, label=label)
        thinkplot.Vlines(ci, -1,2, alpha=0.8, color='0.5')

        print('\n Sample Size: %d' %n)
        print('np.mean(Ls):\n',np.mean(Ls))
        print('ci:\n',ci)
        stderr = RMSE(Ls, 2)
        stderrs.append(stderr)
        print('stderr:\n',stderr)

    thinkplot.Config(xlabel='Sample mean',
                     ylabel='CDF',
                     ylim=[-0.05, 1.05])
    thinkplot.Show()

    ## plot stderrs vs. ns
    thinkplot.Plot(ns, stderrs)
    thinkplot.Show(xlabel='sample size',
                   ylabel='standard error - lambda estimate')
