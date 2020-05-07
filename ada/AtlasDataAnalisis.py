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
    print(bg_roots)
    
    #turn into dfs
    signal_dfs = {signal_root.split('/')[-1][:-5]: root_to_df(signal_root) for signal_root in signal_roots}
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



        