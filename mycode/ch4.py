"""This file contains code used in "Think Stats",
by Ben Smith

Copyright 2020 Ben Smith
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""
## module import
import first
import numpy as np
import thinkstats2
import random
import thinkplot

## function and class definition
def PercentileRank(scores, your_score):
    """Calculates percentile rank of 'your_score' from series of 'scores'
    """
    count = 0
    for score in scores:
        if score <= your_score:
            count += 1

    percentile_rank = 100 * count / len(scores)
    return percentile_rank

def Percentile(scores, percentile_rank):
    """Calculates score at 'percentile_rank' from series of 'scores'
    """
    scores.sort()
    index = percentile_rank * (len(scores)-1) // 100
    return scores[index]

## main scripts
if __name__ == '__main__':
    """Main scripts"""
    ## make frames
    live, firsts, others = first.MakeFrames()

    ## plot cdf
    cdf = thinkstats2.Cdf(live.prglngth, label='prglngth')
    thinkplot.Cdf(cdf)
    thinkplot.Show(xlabel='weeks', ylabel='CDF')

    ## compare firsts and others
    first_cdf = thinkstats2.Cdf(firsts.totalwgt_lb, label='first')
    other_cdf = thinkstats2.Cdf(others.totalwgt_lb, label='others')
    thinkplot.PrePlot(2)
    thinkplot.Cdfs([first_cdf, other_cdf])
    thinkplot.Show(xlabel='weight (pounds)', ylabel='CDF')

    ## perform random sampling and make cdf of sample percentile ranks
    weights = live.totalwgt_lb
    cdf = thinkstats2.Cdf(weights, label='totalwgt_lb')
    sample = np.random.choice(weights, 1000, replace=True)
    ranks = [cdf.PercentileRank(x) for x in sample]

    rank_cdf = thinkstats2.Cdf(ranks)
    thinkplot.Cdf(rank_cdf)
    thinkplot.Show(xlabel='percentile rank', ylabel='CDF')

    ## my birth weight
    my_weight = 8+4/16
    my_rank = first_cdf.PercentileRank(my_weight)
    print('my_rank:\n', my_rank)
    calc_weight = first_cdf.Value(my_rank/100)
    print('calc_weight:\n', calc_weight)

    ## observe random number distribution
    uni = []
    gauss = []
    for i in range(10000):
        uni.append(random.random())
        gauss.append(np.random.normal())

    uni_cdf = thinkstats2.MakeCdfFromList(uni, label='uniform')
    gauss_cdf = thinkstats2.MakeCdfFromList(gauss, label='gauss')

    thinkplot.PrePlot(2)
    thinkplot.Cdf(uni_cdf)
    thinkplot.Cdf(gauss_cdf)
    thinkplot.Show(xlabel='value',ylabel='CDF')

