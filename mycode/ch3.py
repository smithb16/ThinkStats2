## Boilerplate and module import

import nsfg
import thinkstats2
import thinkplot
import numpy as np

## Function and class definition
def BiasPmf(pmf, label):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x,x)

    new_pmf.Normalize()
    return new_pmf

def UnbiasPmf(pmf, label):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x,1.0/x)

    new_pmf.Normalize()
    return new_pmf

def PmfMean(pmf):
    """Calculates mean of pmf.

    pmf: thinkstats2.Pmf
    return: float - mean
    """
    mean = 0
    for x, p in pmf.Items():
        mean += p*x
    return mean

def PmfVar(pmf):
    """Calculates variance of pmf.

    pmf: thinkstats2.Pmf
    return: float - variance
    """
    mean = pmf.Mean()
    var = 0
    for x, p in pmf.Items():
        var += (p * (x-mean)**2)
    return var


if __name__ == '__main__':
    """Main scripts below here."""
    ## Turn pregnancy data into DataFrame
    preg = nsfg.ReadFemPreg()
    resp = nsfg.ReadFemResp()

    ## Make first_pmf and other_pmf
    live = preg[preg.outcome == 1]
    firsts = live[live.birthord == 1]
    others = live[live.birthord != 1]
    first_pmf = thinkstats2.Pmf(firsts.prglngth)
    other_pmf = thinkstats2.Pmf(others.prglngth)

    ## Make comparison plot
    width = 0.45
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(first_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Config(xlabel='weeks',
                     ylabel='probability',
                     axis=[27,46, 0,0.6])

    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([first_pmf, other_pmf])
    thinkplot.Show(xlabel='weeks',
                   axis=[27,46, 0,0.6])

    ## Make bar chart comparison
    diffs = []
    weeks = range(35,46)
    for week in weeks:
        p1 = first_pmf[week]
        p2 = other_pmf[week]
        diff = (p1 - p2) * 100
        diffs.append(diff)

    thinkplot.Bar(weeks, diffs)
    thinkplot.Show(xlabel='weeks',ylabel='diff - %')

    ## Class size paradox
    sizes = range(7,48,5)
    freqs = [8,8,14,4,6,12,8,3,2]
    d = {}
    for freq,size in zip(freqs, sizes):
        d[size] = freq

    pmf = thinkstats2.Pmf(d, label='actual')
    print('mean: ',pmf.Mean())

    bias_pmf = BiasPmf(pmf, label='observed')
    print('biased mean: ',bias_pmf.Mean())

    ## Plot class size paradox
    thinkplot.PrePlot(2)
    thinkplot.Pmfs([pmf, bias_pmf])
    thinkplot.Show(xlabel='class size', ylabel='PMF')

    ## Generate unbiased and biased pmf of kids in family
    pmf = thinkstats2.Pmf(resp.numkdhh, label='actual')
    bias_pmf = BiasPmf(pmf, label='bias')
    print('mean:\n',pmf.Mean())
    print('PmfMean(pmf):\n', PmfMean(pmf))
    print('pmf.Var:\n', pmf.Var())
    print('PmfVar(pmf):\n', PmfVar(pmf))

    #thinkplot.PrePlot(2)
    #thinkplot.Pmfs([pmf, bias_pmf])
    #thinkplot.Show(xlabel='kids in house', ylabel='PMF')

    ## Exercise 3.3
    # measure difference between first and others to the same woman
    d = nsfg.MakePregMap(live)
    resp['prglngth_first'] = np.nan
    resp['prglngth_others'] = np.nan
    for caseid, pregids in d.items():
        if len(pregids) > 0:
            df = live.loc[pregids]
            res = df.loc[df.pregordr==1,'prglngth'].mean()
            resp.prglngth_first[resp.caseid==caseid] = res
        if len(pregids) > 1:
            df = live.loc[pregids]
            res = df.loc[df.pregordr!=1,'prglngth'].mean()
            resp.prglngth_others[resp.caseid==caseid] = res

    ## plot difference between first and others to the same woman
    first_pmf = thinkstats2.Pmf(resp.prglngth_first, label='first')
    other_pmf = thinkstats2.Pmf(np.floor(resp.prglngth_others),label='others')

    width = 0.45
    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(first_pmf, align='right', width=width)
    thinkplot.Hist(other_pmf, align='left', width=width)
    thinkplot.Config(xlabel='weeks',
                     ylabel='probability',
                     axis=[27,46, 0,0.6])

    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([first_pmf, other_pmf])
    thinkplot.Show(xlabel='weeks',
                   axis=[27,46, 0,0.6])

