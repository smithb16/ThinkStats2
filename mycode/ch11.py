## module import

import re
import patsy
import numpy as np
import pandas as pd

import random

import thinkstats2
import thinkplot

import first
import statsmodels.formula.api as smf

## function and class definition
def ReadVariables():
    """Reads Stata dictionary files for NSFG data

    returns: DataFrame that maps variables names to descriptions
    """
    vars1 = thinkstats2.ReadStataDct('2002FemPreg.dct').variables
    vars2 = thinkstats2.ReadStataDct('2002FemResp.dct').variables

    all_vars = vars1.append(vars2)
    all_vars.index = all_vars.name
    return all_vars

def MiningReport(variables, n=30):
    """Prints the variables with the highest R^2.

    t: list of (R^2, variable name) pairs
    n: number of pairs to print
    """
    all_vars = ReadVariables()

    variables.sort(reverse=True)
    relevant = []

    for r2, name in variables[:n]:
        key = re.sub('_r$', '', name)
        try:
            desc = all_vars.loc[key].desc
            if isinstance(desc, pd.Series):
                desc = desc[0]
            print(name, r2, desc)
        except (KeyError, IndexError):
            print(name, r2)
        relevant.append(name)

    return relevant

def GoMiningTotalWgt(df):
    """Search for variables that predict birth weight.

    df: DataFrame of pregnancy records

    return: list of (rsquared, variable name) pairs
    """
    variables = []
    for name in df.columns:
        try:
            if df[name].var() < 1e-7:
                continue

            formula = 'totalwgt_lb ~ agepreg + ' + name

            # The following seems to be required in some environments
            # formula = formula.encode('ascii')

            model = smf.ols(formula, data=df)
            if model.nobs < len(df)/2:
                continue

            results = model.fit()
        except (ValueError, TypeError):
            continue

        variables.append((results.rsquared, name))

    return variables

def GoMiningNumBabes(df):
    """Search for variables that predict number of children borne.

    df: DataFrame of pregnancy records

    return: list of (rsquared, variable name) pairs
    """
    variables = []
    for name in df.columns:
        try:
            if df[name].var() < 1e-7:
                continue

            formula = 'numbabes ~ ' + name

            # The following seems to be required in some environments
            # formula = formula.encode('ascii')

            model = smf.poisson(formula, data=df)
            nobs = len(model.endog)
            if nobs < len(df)/2:
                continue

            results = model.fit()
        except (ValueError, TypeError):
            continue

        variables.append((results.prsquared, name))

    return variables

def GoMiningPrgLngth(df):
    """Search for variables that predict sex.

    df: DataFrame of pregnancy records

    return: list of (rsquared, variable name) pairs
    """
    variables = []
    for name in df.columns:
        try:
            if df[name].var() < 1e-7:
                continue

            formula = 'prglngth ~ ' + name

            # The following seems to be required in some environments
            # formula = formula.encode('ascii')

            model = smf.ols(formula, data=df)
            if model.nobs < len(df)/2:
                continue

            results = model.fit()
        except (ValueError, TypeError):
            continue

        variables.append((results.rsquared, name))

    return variables

def GoMiningSexRatio(df):
    """Search for variables that predict sex.

    df: DataFrame of pregnancy records

    return: list of (rsquared, variable name) pairs
    """
    df['boy'] = (df.babysex==1).astype(int)

    variables = []
    for name in df.columns:
        try:
            if df[name].var() < 1e-7:
                continue

            formula = 'boy ~ agepreg + ' + name
            model = smf.logit(formula, data=df)
            nobs = len(model.endog)
            if nobs < len(df)/2:
                continue

            results = model.fit()
        except:
            continue

        variables.append((results.prsquared, name))

    return variables

## main scripts
if __name__ == '__main__':
    """main scripts"""
    ## make nsfg dataframes
    live, firsts, others = first.MakeFrames()

    ## calculate age birth weight regression
    formula = 'totalwgt_lb ~ agepreg'
    model = smf.ols(formula, data=live)
    results = model.fit()
    results.summary()

    ## extract results
    inter = results.params['Intercept']
    slope = results.params['agepreg']
    print('inter, slope:\n',inter, slope)

    slope_pvalue = results.pvalues['agepreg']
    print('slope_pvalue:\n',slope_pvalue)

    print('results.rsquared:\n',results.rsquared)

    ## split first babies from live and perform regression
    live['isfirst'] = live.birthord == 1
    formula = 'totalwgt_lb ~ isfirst'
    results = smf.ols(formula, data=live).fit()
    results.summary()

    ## multiple regression
    formula = 'totalwgt_lb ~ isfirst + agepreg'
    results = smf.ols(formula, data=live).fit()
    results.summary()

    ## add agepreg**2 to regression
    live['agepreg2'] = live.agepreg**2
    formula = 'totalwgt_lb ~ isfirst + agepreg + agepreg2'
    results = smf.ols(formula, data=live).fit()
    results.summary()

    ## data mining - make joined database
    import nsfg

    live = live[live.prglngth>30]
    resp = nsfg.ReadFemResp()
    resp.index = resp.caseid
    join = live.join(resp, on='caseid', rsuffix='_r')

    ## mine the join DataFrame
    variables = GoMiningTotalWgt(join)

    ## read variables
    MiningReport(variables)

    ## new formula to win office betting pool
    formula = ('totalwgt_lb ~ agepreg + C(race) + babysex==1 + '
               'nbrnaliv>1 + paydu==1 + totincr')
    results = smf.ols(formula, data=join).fit()
    results.summary()

    ## logistic regressions
    y = np.array([0,1,0,1])
    x1 = np.array([0,0,0,1])
    x2 = np.array([0,1,1,1])

    beta = [-1.5, 2.8, 1.1]

    log_o = beta[0] + beta[1] * x1 + beta[2] * x2
    print('log_o:\n',log_o)
    o = np.exp(log_o)
    print('odds:\n',o)
    p = o / (o+1)
    print('probabilities:\n',p)
    likes = np.where(y, p, 1-p)
    print('likes:\n',likes)
    like = np.prod(likes)
    print('likelihood:\n',like)

    ## perform logistic regression of sex ratio
    import first
    live, irsts, others = first.MakeFrames()
    live = live[live.prglngth>30]
    live['boy'] = (live.babysex==1).astype(int)

    ## test models
    model = smf.logit('boy ~ agepreg + hpagelb + birthord + C(race)', data=live)
    results = model.fit()
    results.summary()

    ## calculate baseline correct
    endog = pd.DataFrame(model.endog, columns=[model.endog_names])
    exog = pd.DataFrame(model.exog, columns=[model.exog_names])

    actual = endog['boy']
    baseline = actual.mean()
    print('baseline:\n',baseline)

    ## count number we get correct
    predict = (results.predict() >= 0.5)
    true_pos = predict * actual
    true_neg = (1 - predict) * (1 - actual)
    sum(true_pos), sum(true_neg)

    ## calculate accuracy vs baseline
    print('baseline:\n',baseline)
    acc = (sum(true_pos) + sum(true_neg)) / len(actual)
    print('acc:\n',acc)

    ## make predictions for onen individual
    columns = ['agepreg', 'hpagelb', 'birthord', 'race']
    new = pd.DataFrame([[35, 39, 3, 2]], columns=columns)
    y = results.predict(new)
    print('y:\n',y)

    ############################################
    ## exercises
    ############################################

    # predict prglngth
    live = live[live.prglngth>30]
    live['boy'] = (live.babysex==1).astype(int)
    resp = nsfg.ReadFemResp()
    resp.index = resp.caseid
    join = live.join(resp, on='caseid', rsuffix='_r')
    join.screentime = pd.to_datetime(join.screentime)

    ## mine the join DataFrame
    variables = GoMiningPrgLngth(join)

    ## read variables
    relevant = MiningReport(variables, n=100)
    print('birthord in relevant:\n','birthord' in relevant)
    print('race in relevant:\n','race' in relevant)

    ## make model with relevant predictive variables
    model = smf.ols('prglngth ~ birthord==1 + race==2 + nbrnaliv > 1', data=live)
    results = model.fit()
    results.summary()

    ## predict baby sex
    # make join DataFrame
    live = live[live.prglngth>30]
    live['boy'] = (live.babysex==1).astype(int)
    resp = nsfg.ReadFemResp()
    resp.index = resp.caseid
    join = live.join(resp, on='caseid', rsuffix='_r')
    join.screentime = pandas.to_datetime(join.screentime)

    ## mine the join DataFrame
    variables = GoMiningSexRatio(join)

    ## read variables
    relevant = MiningReport(variables)

    ## make model with relevant predictive variables
    model = smf.logit('boy ~ totalwgt_lb + agepreg + lbw1 + fmarout5', data=join)
    results = model.fit()
    results.summary()

    ## calculate baseline correct
    endog = pd.DataFrame(model.endog, columns=[model.endog_names])
    exog = pd.DataFrame(model.exog, columns=[model.exog_names])

    actual = endog['boy']
    baseline = actual.mean()
    print('baseline:\n',baseline)

    ## count number we get correct
    predict = (results.predict() >= 0.5)
    true_pos = predict * actual
    true_neg = (1 - predict) * (1 - actual)
    sum(true_pos), sum(true_neg)

    ## calculate accuracy vs baseline
    print('baseline:\n',baseline)
    acc = (sum(true_pos) + sum(true_neg)) / len(actual)
    print('acc:\n',acc)

    ## mine the join DataFrame
    variables = GoMiningNumBabes(join)

    ## read the variables
    relevant = MiningReport(variables, n=60)

    ## make model with relevant predictive variables
    model = smf.poisson('numbabes ~ ager + educat + C(race) + totincr', data=join)
    results = model.fit()
    results.summary()

    ## predict numbabes
    columns = ['ager','race','educat','totincr']
    new = pd.DataFrame([[35,1,16,14]], columns=columns)
    predict_babes = results.predict(new)
    print('predict_babes:\n',predict_babes)

    ## predict married/divorced
    model = smf.mnlogit('rmarital ~ ager + C(race) + totincr + educat', data=join)
    results = model.fit()
    results.summary()

    ## test individual
    new = pd.DataFrame({'ager': [25],
                       'race': [2],
                       'totincr': [11],
                       'educat': [12]})
    predict = results.predict(new)
    print('predict:\n',predict)
