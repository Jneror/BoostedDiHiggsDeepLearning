import ROOT
import pandas as pd
import numpy as np
from root_numpy import tree2array
from os import path
from glob import glob
import json

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from ada2 import col_names, selected_features, tags

########################
### Process datasets ###
########################

def filter_region(df, region):
    """Filter dataframe by m_region"""
    return df[df["m_region"] == region]

def filter_tag(df, tag):
    """Filter dataframe by m_FJNbtagJets"""
    return df[df["m_FJNbtagJets"] == tag]

#added on Jun 2020
def filter_sample(df, sample):
    """Filter dataframe by sample"""
    return df[df["sample"] == sample]

def drop_fakes(df):
    """Return df without fake samples"""
    #display(df[(df["sample"] == "fakes") & (df["EventWeight"] > 0)])
    return df[df["sample"] != "fakes"].reset_index(drop = True)

def drop_twodim(df):
    """Return df without TwoDimMassWindow Region events"""
    return df[df["m_region"] != "TwoDimMassWindow"].reset_index(drop = True)

#added on Jun 2020
def classify(signal_dfs, bg_df, class_col):
    #signal labeled as class 1
    for i in range(len(signal_dfs)):
        signal_dfs[i][class_col] = 1

    #bg labeled as class 0
    bg_df[class_col] = 0

    df = pd.concat(signal_dfs + [bg_df], axis = 0)
    return df

#updated on Jun 2020
def preprocess_df(df, signal, region, tag):
    """Preprocess: drop two dim, filter region if any, filter tag if any and classify signals"""
    if region is not None: df = filter_region(df, region)
    if tag is not None: df = filter_tag(df, tag)
    return df.reset_index(drop = True)

#added on jun 2020
def read_dataset(source_path, signal, bg, region, tag):
    signal_df = pd.read_csv(f"{source_path}/{signal}.csv")
    bg_df = pd.read_csv(f"{source_path}/{bg}.csv")

    df = classify([signal_df], bg_df, "label")
    df = preprocess_df(df, signal, region, tag)

    cols_to_pop = [col_names["weight"], "label"]
    if tag is None: cols_to_pop.append(col_names["tag"])
    df = df[cols_to_pop + selected_features]
    return df

####################
### Gen Datasets ###
####################

def root_to_df(file):
    """Turns root file into a dataframe"""
    rfile = ROOT.TFile(file) #root file
    intree = rfile.Get("Nominal") #get tree from root file
    return pd.DataFrame(tree2array(intree)) #DataFrame from array from root file

#updated on Jun 2020 
def gen_signal_datasets(signal, source_path, dest_path):
    #get roots
    all_roots = glob(f"{source_path}/*.root")
    signal_roots = glob(f"{source_path}/{signal}*.root")
    data_roots = glob(f"{source_path}/data*.root")
    bg_roots = list(set(all_roots) - set(signal_roots) - set(data_roots))
    
    #turn into dfs, drop fakes in signal, but not in bg, drop two dim in all
    signal_dfs = {root.split('/')[-1][:-5]: drop_fakes(drop_twodim(root_to_df(root))) for root in signal_roots}
    bg_dfs = [drop_twodim(root_to_df(root)) for root in bg_roots]
    data_fakes_df = [filter_sample(drop_twodim(root_to_df(root)), "fakes") for root in data_roots]
    
    #join all backgrounds
    bg_df = pd.concat(bg_dfs + data_fakes_df, axis = 0)
    
    #save background
    bg_df.to_csv(f"{dest_path}/{signal}_background.csv", index = False)
    
    #save signals
    for signal_name, signal_df in signal_dfs.items():
        signal_df.to_csv(f"{dest_path}/{signal_name}.csv", index = False)


#################
### Training ####
#################

def pop_col_from_dfs(dfs, col):
    """Drop the column from a list of dfs and return this columns in form of array"""
    popped_cols = [df.pop(col).values for df in dfs]
    return popped_cols

def standar_scale_dfs(dfs):
    scaler = StandardScaler().fit(dfs[0])
    return [pd.DataFrame(scaler.transform(df),columns = df.columns) for df in dfs]

def rotate_vectors(vectors):
    return [v.reshape(-1, 1) for v in vectors]

def trainvaltest_split(x, y, seed, train_size, val_size, test_size):
    """Split x and y into train, val and test datasets"""
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x,
        y,
        test_size = test_size,
        shuffle = True,
        random_state = seed
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp,
        y_tmp,
        test_size = val_size/(test_size + train_size),
        shuffle = True,
        random_state = seed
    )
    return x_train, x_val, x_test, y_train, y_val, y_test

def trainvaltest_split_bytag(x, y, seed, train_size, val_size, test_size):

    x_tag = {tag: x[x[col_names["tag"]] == tag] for tag in tags}
    y_tag = {tag: y[x[col_names["tag"]] == tag] for tag in tags}

    x_train, x_val, x_test, y_train, y_val, y_test = ({},{},{},{},{},{})

    for t in tags:
        x_train[t], x_val[t], x_test[t], y_train[t], y_val[t], y_test[t] = trainvaltest_split(
            x_tag[t], y_tag[t], seed, train_size, val_size, test_size
        )

    x_train, y_train = shuffle(pd.concat(x_train.values()),np.concatenate(list(y_train.values())),random_state=seed)
    x_val, y_val = shuffle(pd.concat(x_val.values()),np.concatenate(list(y_val.values())),random_state=seed)
    x_test, y_test =shuffle(pd.concat(x_test.values()),np.concatenate(list(y_test.values())),random_state=seed)

    return (x_train.reset_index(drop=True), x_val.reset_index(drop=True), x_test.reset_index(drop=True),
    y_train, y_val, y_test)

#added on ago 2020
def trainvaltest_split_bymass(x, y, seed, train_size, val_size, test_size, masses):
    x_mass = {mass: x[x["mass"] == mass] for mass in masses}
    y_mass = {mass: y[x["mass"] == mass] for mass in masses}

    x_train, x_val, x_test, y_train, y_val, y_test = ({},{},{},{},{},{})

    for t in masses:
        x_train[t], x_val[t], x_test[t], y_train[t], y_val[t], y_test[t] = trainvaltest_split(
            x_mass[t], y_mass[t], seed, train_size, val_size, test_size
        )
    x_train, y_train = shuffle(pd.concat(x_train.values()),np.concatenate(list(y_train.values())),random_state=seed)
    x_val, y_val = shuffle(pd.concat(x_val.values()),np.concatenate(list(y_val.values())),random_state=seed)
    x_test, y_test =shuffle(pd.concat(x_test.values()),np.concatenate(list(y_test.values())),random_state=seed)

    return (x_train.reset_index(drop=True), x_val.reset_index(drop=True), x_test.reset_index(drop=True),
    y_train, y_val, y_test)

#added on jun 2020
def split_dataset(df, train_size, val_size, test_size, seed):

    x = df.drop(columns = ["label"])
    y = df["label"].values

    #object where all the datasets will be stored
    sets = {}

    #split into train, val and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = trainvaltest_split(x, y, seed, train_size, val_size, test_size)

    #all the sets for train, val and test will be stored here
    w_train, w_val, w_test = pop_col_from_dfs([x_train, x_val, x_test], col_names["weight"])
    sets["w"] = {"train": w_train, "val": w_val, "test": w_test}

    #scale
    x_train, x_val, x_test = standar_scale_dfs([x_train, x_val, x_test])
    sets["x"] = {"train": x_train, "val": x_val, "test": x_test}

    #reshape y
    y_train, y_val, y_test = rotate_vectors([y_train, y_val, y_test])
    sets["y"] = {"train": y_train, "val": y_val, "test": y_test}

    return sets

def split_dataset_bytag(df, train_size, val_size, test_size, seed):

    x = df.drop(columns = ["label"])
    y = df["label"].values

    #object where all the datasets will be stored
    sets = {}

    #split into train, val and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = trainvaltest_split_bytag(x,y,seed,train_size,val_size,test_size)
    x_test_bytag = {tag: x_test[x_test[col_names["tag"]] == tag]  for tag in tags}
    y_test_bytag = {tag: y_test[x_test[col_names["tag"]] == tag]  for tag in tags}

    #all the sets for train, val and test will be stored here
    w_train, w_val, w_test = pop_col_from_dfs([x_train, x_val, x_test], col_names["weight"])
    sets["w"] = {"train": w_train, "val": w_val, "test": w_test}

    #scale
    scaler = StandardScaler().fit(x_train)
    x_train, x_val, x_test = [pd.DataFrame(scaler.transform(df),columns=df.columns) for df in [x_train, x_val, x_test]]
    sets["x"] = {"train": x_train, "val": x_val, "test": x_test}

    #reshape y
    y_train, y_val, y_test = rotate_vectors([y_train, y_val, y_test])
    sets["y"] = {"train": y_train, "val": y_val, "test": y_test}

    for tag in tags:
        sets[tag] = {}
        x_test, y_test = (x_test_bytag[tag], y_test_bytag[tag])
        sets[tag]["w_test"] = pop_col_from_dfs([x_test], col_names["weight"])[0]
        sets[tag]["x_test"] = pd.DataFrame(scaler.transform(x_test),columns=x_test.columns)
        sets[tag]["y_test"] = rotate_vectors([y_test])[0]

    return sets

#added on ago 2020
def split_dataset_bymass(df, train_size, val_size, test_size, seed, masses):
    x = df.drop(columns = ["label"])
    y = df["label"].values

    x_train, x_val, x_test, y_train, y_val, y_test = trainvaltest_split_bymass(x,y,seed,train_size,val_size,test_size,masses)
    x_test_bymass = {mass: x_test[x_test["mass"] == mass]  for mass in masses}
    y_test_bymass = {mass: y_test[x_test["mass"] == mass]  for mass in masses}

    #object where all the datasets will be stored
    sets = {}

    #all the sets for train, val and test will be stored here
    w_train, w_val, w_test = pop_col_from_dfs([x_train, x_val, x_test], col_names["weight"])
    sets["w"] = {"train": w_train, "val": w_val, "test": w_test}

    #scale
    scaler = StandardScaler().fit(x_train)
    x_train, x_val, x_test = [pd.DataFrame(scaler.transform(df),columns=df.columns) for df in [x_train, x_val, x_test]]
    sets["x"] = {"train": x_train, "val": x_val, "test": x_test}

    #reshape y
    y_train, y_val, y_test = rotate_vectors([y_train, y_val, y_test])
    sets["y"] = {"train": y_train, "val": y_val, "test": y_test}

    for mass in masses:
        sets[mass] = {}
        x_test, y_test = (x_test_bymass[mass], y_test_bymass[mass])
        sets[mass]["w_test"] = pop_col_from_dfs([x_test], col_names["weight"])[0]
        sets[mass]["x_test"] = pd.DataFrame(scaler.transform(x_test),columns=x_test.columns)
        sets[mass]["y_test"] = rotate_vectors([y_test])[0]

    return sets