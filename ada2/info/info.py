import pandas as pd

def wavg(df, x_col, weight_col):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    x * w / sum(w)
    """
    x = df[x_col]
    w = df[weight_col]
    try:
        return (x * w).sum() / w.sum()
    except ZeroDivisionError:
        return x.mean()

def signal_density(df, signal_col):
    """Returns the class 1 count divided by the total count (class 1 and class 0)"""
    classes = df[signal_col]
    return classes.sum()/classes.count()

def signal_count(df, signal_col):
    """Returns the amount of signal (class 1) in a dataset"""
    return df[signal_col].sum()

def total_count(df):
    """Returns the amount of data in a dataset"""
    return df.shape[0]

def signal_distribution(df, signal_col, weight_col):
    """Returns info about the signal (class 1) in a dataset"""
    return pd.Series(
        [
            wavg(df, signal_col, weight_col),
            signal_density(df, signal_col),
            signal_count(df, signal_col),
            total_count(df),
        ],
        index = ["signal w-density", "signal density", "signal count", "total count"]
    )

def signal_distribution_per(df, cols, signal_col = "signal", weight_col = "EventWeight"):
    """Returns density information about the signal per every feature in cols"""
    if cols != []:
        return df.groupby(cols).apply(signal_distribution, signal_col, weight_col)
    return df.apply(signal_distribution, signal_col, weight_col)