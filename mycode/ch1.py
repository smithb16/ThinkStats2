## Boilerplate and module import

import nsfg
import numpy as np

if __name__ == '__main__':
    """Main scripts"""
    ## Turn pregnancy data into DataFrame
    preg = nsfg.ReadFemPreg()
    resp = nsfg.ReadFemResp()

    ## Make plots
    #ages = np.floor(live.agepreg)
    #hist = thinkstats2.Hist(ages, label='agepreg')
    #thinkplot.Hist(hist)
    #thinkplot.Show(xlabel='years', ylabel='count')

    length = np.floor(live.prglngth)
    hist = thinkstats2.Hist(length, label='prglngth')
    thinkplot.Hist(hist)
    thinkplot.Show(xlabel='weeks', ylabel='count')

    ## Work with preg
    t = preg.birthord.value_counts().sort_index()
    lg = preg.prglngth.value_counts().sort_index()
    na = preg.birthord.isnull().sum()
    #print('pregnancy length:\n', lg)

    short_preg = preg.loc[preg.prglngth < 15, 'outcome']
    #print('short_preg:\n', short_preg)
    #print('short_preg outcomes:\n',short_preg.value_counts().sort_index())

    #print('mean totalwgt_lb:\n', preg.totalwgt_lb.mean())
    #print('mean totalwgt_kg:\n', preg.totalwgt_kg.mean())

    #print('age range:', min(resp.age_r), max(resp.age_r))
    res = resp.pregnum.value_counts().sort_index()
    pt = resp.caseid[resp.pregnum==19].values[0]
    pt_preg = preg.outcome[preg.caseid==pt]
    outs = preg.agepreg[preg.caseid==pt]
    #print('pt:\n', pt)
    #print('pt_preg:\n', pt_preg)
    #print('outs:\n', outs)
    #print('pregnancy numbers :\n', res )
