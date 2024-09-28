import pandas as pd
import numpy as np
import scipy.stats
pd.options.display.float_format = '{:.4f}'.format

def sharpe_ratio(rets:pd.DataFrame,rfr=0.03):
    if isinstance(rets,pd.Series):
        rets=rets.to_frame()
    Annualized_rets=((1+rets).prod())**(12/rets.shape[0])-1
    Annualized_volatility=rets.std()*np.sqrt(12)
    sharpe_ratio=(Annualized_rets-rfr)/Annualized_volatility
    return pd.DataFrame({"Annualized Returns":Annualized_rets,
                         "Annualized Volatility":Annualized_volatility,
                         "Sharpe Ratio":sharpe_ratio})

def drawdowns(rets:pd.Series,init_inv=1000):
    wealth_index=init_inv*((1+rets).cumprod())
    last_peak=wealth_index.cummax()
    drawdown=(wealth_index-last_peak)/last_peak
    return pd.DataFrame({"Wealth Index":wealth_index,
                         "Last Peak":last_peak,
                         "Drawdown":drawdown})

def magic_moments(rets:pd.DataFrame,moment):
    """
    An Alternate method to find skewness and kurtosis
    You can always use sci py stats methods like skew() and kurtosis()
    for skewness pass moment=3
    for kurtosis pass moment=4
    it wont give the excess kurtosis
    """
    if isinstance(rets,pd.Series):
        rets=rets.to_frame()
    demeaned_rets=rets-rets.mean()
    exp=(((demeaned_rets)**moment).mean())/(rets.std(ddof=0))**moment
    return exp

def is_normal(rets:pd.DataFrame,level=0.1):
    """
    Applying jarque-bera test and checking normal or not for the p-value of levels,default=0.1
    """
    statistics,p_value=scipy.stats.jarque_bera(rets)
    return p_value>level

def semi_deviation(rets:pd.DataFrame):
    neg_rets=rets<0
    return rets[neg_rets].std(ddof=0)

def var_historic(rets,level=5):
    """
    Estimate VaR historic for certain percent of level
    ie, returns fall below this for level percent of time
    or, returns are above for 100-level percent of the time
    """
    if isinstance(rets,pd.DataFrame):
        return rets.aggregate(var_historic,level=level)
    elif isinstance(rets,pd.Series):
        return -np.percentile(rets,level)
    else:
        raise TypeError("Expected a series or Dataframe")
        
def var_assumption(rets:pd.DataFrame,level=5,modified=False):
    """
    Estimating VaR gaussian or cornish-fisher for the given percent of risk
    """
    from scipy.stats import norm
    z_scr=norm.ppf(level/100)
    if modified:
        s=magic_moments(rets,3)
        k=magic_moments(rets,4)
        z_scr=(z_scr +
                (z_scr**2 - 1)*s/6 +
                (z_scr**3 -3*z_scr)*(k-3)/24 -
                (2*z_scr**3 - 5*z_scr)*(s**2)/36
              )
               
    return -(rets.mean()+z_scr*rets.std(ddof=0))

def historic_cvar(rets,level=5):
    if isinstance(rets,pd.Series):
        is_beyond=rets<-(var_historic(rets,level=level))
        return rets[is_beyond].mean()
    elif isinstance(rets,pd.DataFrame):
        return rets.aggregate(historic_cvar,level=level)
    else:
        raise TypeError("Expected a series or Dataframe")