import ROOT
import pandas as pd
from root_numpy import tree2array
from os import path
from glob import glob

####################
### Gen Datasets ###
####################

def root_to_df(file):
    """Turns root file into a dataframe"""
    rfile = ROOT.TFile(file) #root file
    intree = rfile.Get("Nominal") #get tree from root file
    return pd.DataFrame(tree2array(intree)) #DataFrame from array from root file

def gen_signal_datasets(signal, data_path, prodata_path):
    #get roots
    all_roots = glob(f"{data_path}/*.root")
    signal_roots = glob(f"{data_path}/{signal}*.root")
    bg_roots = list(set(all_roots) - set(signal_roots) - set(glob(f"{data_path}/data.root")))
    
    #turn into dfs
    #drop fakes in signal, but not in bg
    signal_dfs = {signal_root.split('/')[-1][:-5]: drop_fakes(root_to_df(signal_root)) for signal_root in signal_roots}
    bg_dfs = [root_to_df(bg_root) for bg_root in bg_roots]
    
    #join all backgrounds
    bg_df = pd.concat(bg_dfs, axis = 0)
    
    #join background with the signals
    datasets = {sign_name: pd.concat([sign_df, bg_df], axis = 0) for sign_name, sign_df in signal_dfs.items()}
    
    #save datasets into csvs
    for signal_name, signal_df in datasets.items():
        signal_df.to_csv(f"{prodata_path}/{signal_name}.csv", index = False)
        
########################
### Process datasets ###
########################

def filter_region(df, region):
    """Filter dataframe by m_region"""
    return df[df["m_region"] == region]

def filter_tag(df, tag):
    """Filter dataframe by m_FJNbtagJets"""
    return df[df["m_FJNbtagJets"] == tag]

#signal shouldnt have fakes uwu
def drop_fakes(df):
    """Return df without fake samples"""
    #display(df[(df["sample"] == "fakes") & (df["EventWeight"] > 0)])
    return df[df["sample"] != "fakes"].reset_index(drop = True)

def drop_twodim(df):
    """Return df without TwoDimMassWindow Region events"""
    return df[df["m_region"] != "TwoDimMassWindow"].reset_index(drop = True)

def classify_events(df, signal, class_column):
    df[class_column] = (df["sample"].str.contains(f".*{signal}[^0-9]*",regex = True)).astype(int)
    return df

#####################
### Info datasets ###
#####################

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
    return df.groupby(cols).apply(signal_distribution, signal_col, weight_col)



        