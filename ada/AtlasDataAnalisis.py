import ROOT
import pandas as pd
from root_numpy import tree2array
from os import path
from glob import glob
import matplotlib.pyplot as plt
#dnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import model_from_json
#abstract classes
from abc import ABC, abstractmethod

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


#################
### Training ####
#################

def pop_col(X, col):
    """Returns the popped column with the new dataframe"""
    #pandas has a pop column but modifies the original dataframe, which i dont want
    return X[col], X.drop(columns = [col])

def train_val_test_split(df_X, df_y, train_size, val_size, test_size,
    pop_w = True, w_col = "EventWeight", seed = 1):

    """Split dataframe into train, test and val datasets.
    If pop_w is True, then the weights column is separated from the X datasets"""

    if train_size + val_size + test_size != 1.0:
        print("Incorrect sizes!")
        return None
    
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        df_X,
        df_y,
        test_size = test_size,
        shuffle = True,
        random_state = seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size = val_size/(test_size + train_size),
        shuffle = True,
        random_state = seed
    )

    if pop_w:
        w_train, X_train = pop_col(X_train, w_col)
        w_val, X_val = pop_col(X_val, w_col)
        w_test, X_test = pop_col(X_test, w_col)
        return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_trainvaltest_from_dataset(data_path, signal, region = "SR", tag = 1,
    train_size = 0.6, val_size = 0.2, test_size = 0.2, seed = 1):

    if region not in {"SR", "QCDCR"}:
        print("Error: Region not valid!")
        return
    
    if tag not in {0, 1, 2}:
        print("Error: Tag not valid")
        return
    
    if train_size + val_size + test_size != 1.0:
        print("Error: Datasets sizes dont add 1")
        return

    if not path.exists(f"{data_path}/{signal}.csv"):
        print("Error: Dataset not found")
        return

    #import dataset
    data = pd.read_csv(f"{data_path}/{signal}.csv")

    #drop two dim, filter region and tag
    data = filter_tag(filter_region(drop_twodim(data), region), tag)

    #classify (label)
    data = classify_events(data, signal, "label")

    #features selected by domain expert
    selected_features = ['m_FJpt', 'm_FJeta', 'm_FJphi', 'm_FJm', 'm_DTpt', 'm_DTeta', 'm_DTphi', 'm_DTm',
    'm_dPhiFTwDT', 'm_dRFJwDT', 'm_dPhiDTwMET', 'm_MET', 'm_hhm', 'm_bbttpt', "label"]

    data = data[["EventWeight"] + selected_features]

    df_X = data.drop(columns = ["label"])
    df_y = data["label"]

    #split dataset
    X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test = train_val_test_split(
        df_X,
        df_y,
        train_size, #train
        val_size, #val
        test_size, #test
        seed = seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
    X_val = pd.DataFrame(scaler.transform(X_val),columns=X_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

    #reshape ys and ws
    y_train = y_train.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    w_train = w_train.values
    w_val = w_val.values
    w_test = w_test.values

    return X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test