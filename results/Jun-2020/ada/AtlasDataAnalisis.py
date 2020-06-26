#################
### Libraries ###
#################

import ROOT
import pandas as pd
import numpy as np
from root_numpy import tree2array
from os import path
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import json
#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, roc_auc_score,
precision_recall_curve, average_precision_score)
from sklearn.utils import shuffle
#keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import adagrad, adam, adamax
from keras.models import Model
from keras.models import model_from_json
#abstract
from abc import ABC, abstractmethod

##############
### Config ###
##############

col_names = {
    "region": "m_region",
    "tag": "m_FJNbtagJets",
    "weight": "EventWeight",
}

selected_features = [
    'm_FJpt', 'm_FJeta', 'm_FJphi', 'm_FJm', 'm_DTpt', 'm_DTeta', 'm_DTphi', 'm_DTm',
    'm_dPhiFTwDT', 'm_dRFJwDT', 'm_dPhiDTwMET', 'm_MET', 'm_hhm', 'm_bbttpt',
]

tags = [0, 1, 2]

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
    if cols != []:
        return df.groupby(cols).apply(signal_distribution, signal_col, weight_col)
    return df.apply(signal_distribution, signal_col, weight_col)


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

################
### Plotting ###
################

def plot_confidence_matrix(arr, width, height, fmt):
    df_cm = pd.DataFrame(arr, index = range(2), columns = range(2))
    plt.figure(figsize=(width, height))
    sns.heatmap(df_cm, annot=True, fmt = fmt)
    plt.title("Confusion matrix")
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.show()

def plot_train_loss(train_loss, val_loss, epochs, width = 10, height = 6):
    plt.figure(1, figsize=(width, height))
    plt.plot(range(epochs), train_loss)
    plt.plot(range(epochs), val_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend(["Training", "Validation"])
    plt.show()

def plot_train_acc(train_acc, val_acc, epochs, width = 10, height = 6):
    plt.figure(1, figsize=(width, height))
    plt.plot(range(epochs), train_acc)
    plt.plot(range(epochs), val_acc)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Training vs Validation Accuracy")
    plt.grid(True)
    plt.legend(["Training", "Validation"])
    plt.show()

def plot_test_roc(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label="{} (area = {:.3f})".format(model_name, roc_auc))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def plot_test_recall(precision, recall, prec_rec_auc, model_name):
    plt.figure()
    plt.plot([0, 1], [1, 0], 'k--')
    plt.plot(recall, precision, label='{} (area = {:.3f})'.format(model_name, prec_rec_auc))
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-recall curve')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

############################
### Model Mother Classes ###
############################

class KerasModel(ABC):
    """Mother class for keras models"""
    def __init__(self, n_input):
        self.model = None
        self.history = None
        self.title = ""
        self.model_name = ""
    
    def fit(self, X_train, y_train, w_train, X_val, y_val, w_val, epochs):
        if self.model is not None:
            self.history = pd.DataFrame(self.model.fit(
                X_train.values, y_train, sample_weight = w_train,
                epochs = epochs,
                verbose = 1,
                validation_data = (X_val.values, y_val, w_val)
            ).history)
        else:
            print("Build your model first!")
    
    def plot_loss(self, width = 10, height = 6):
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        plot_train_loss(loss, val_loss, len(loss), width = 10, height = 6)
    
    def save(self, directory, version):
        if self.model is not None:
            print("modelo")
            #save model
            model_json = self.model.to_json()
            with open(f"{directory}/{self.model_name}_{version}.json", "w") as json_file:
                json_file.write(model_json)
                print("modelo")
            #save weights
            self.model.save_weights(f"{directory}/{self.model_name}_{version}.h5")
        if self.history is not None:
            print("historia")
            #save training data
            with open(f"{directory}/{self.model_name}_{version}.csv", mode='w') as f:
                print("mas historia")
                self.history.to_csv(f)
    
    def load(self, directory, version, model_name = None):
        if model_name is not None:
            self.model_name = model_name
        # load json and create model
        if self.model is None:
            json_file = open(f"{directory}/{self.model_name}_{version}.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
        #load weights
        self.model.load_weights(f"{directory}/{self.model_name}_{version}.h5")
        #load train history
        self.history = pd.read_csv(f"{directory}/{self.model_name}_{version}.csv")
    
    #added on Jun 2020
    @abstractmethod
    def predict(self, x_test, th):
        print("Not implemented :(")
    
    #added on Jun 2020
    def complete_evaluation(self, x_test, y_test, w_test, th, save = False, dest_path = ".", name = "v1"):
        #prediction
        y_pred = self.predict(x_test, th)

        #class report
        class_report = classification_report(y_test, y_pred, output_dict = True)
        weighted_class_report = classification_report(y_test, y_pred, output_dict = True, sample_weight=w_test)

        #accuracy
        acc = class_report["accuracy"]
        del class_report["accuracy"]
        weighted_acc = weighted_class_report["accuracy"]
        del weighted_class_report["accuracy"]

        #confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        weighted_cm = confusion_matrix(y_test, y_pred, sample_weight=w_test).tolist()

        #all the evaluations
        complete_eval = {
            "class_report": class_report,
            "weighted_class_report": weighted_class_report,
            "accuracy": acc,
            "weighted_accuracy": weighted_acc,
            "cm": cm,
            "weighted_cm": weighted_cm,
        }

        #save evaluation
        if save:
            json_eval = json.dumps(complete_eval)
            with open(f"{dest_path}/eval_{self.model_name}_{name}.json", 'w') as json_file:
                json_file.write(json_eval)

        return complete_eval

class Autoencoder(KerasModel):
    def __init__(self, n_input, anomaly_class):
        super().__init__(n_input)
        self.anomaly_class = anomaly_class

    def fit(self, x_train, w_train, x_val, w_val, epochs):
        super().fit(x_train, x_train, w_train, x_val, x_val, w_val, epochs)
    
    #added on Jun 2020
    def plot_reconstruction_error(self, x_test, y_test, th):
        x_pred = self.model.predict(x_test)
        errors = np.mean(np.power(x_test.values - x_pred, 2), axis=1)
        color = ListedColormap(["#1f85ad", "#ff7b00"])
        plt.figure(figsize=(14, 10))
        plt.scatter(x = range(len(errors)), y = errors, c = y_test.flatten(), s=16, cmap = color)
        plt.axhline(y=th, color='red', linestyle='-')
        plt.show()
    
    #added on Jun 2020
    def predict(self, x_test, th):
        x_pred = self.model.predict(x_test)
        errors = np.mean(np.power(x_test.values - x_pred, 2), axis=1)
        
        if self.anomaly_class == 1:
            y_pred = (errors > th).astype(int)
        else:
            y_pred = (errors <= th).astype(int)
            
        return y_pred
    
    #added on Jun 2020
    def plot_confidence_matrix(self, x_test, y_test, th, fmt):
        #prediction
        y_pred = self.predict(x_test, th)
        #confidence matrix
        conf_matrix = confusion_matrix(y_test.flatten(), y_pred)
        #plot
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, fmt=fmt)
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()

class BinaryClassifier(KerasModel):

    def __init__(self, n_input):
        super().__init__(n_input)

    def evaluate_with_weights(self, X_test, y_test, w_test, threshold = 0.4):
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > threshold)

        print("Classification Report")
        print(classification_report(y_test, y_pred, sample_weight = w_test))
        print("Confussion Matrix")
        print(confusion_matrix(y_test, y_pred, sample_weight = w_test))
    
    def evaluate(self, X_test, y_test, threshold = 0.4):
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > threshold)

        print("Classification Report")
        print(classification_report(y_test, y_pred))
        print("Confussion Matrix")
        print(confusion_matrix(y_test, y_pred))

    def plot_roc(self, X_test, y_test):
        y_scores = self.model.predict(X_test).ravel()
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = roc_auc_score(y_test, y_scores)
        plot_test_roc(fpr, tpr, roc_auc, model_name)
    
    def plot_recall(self, X_test, y_test):
        y_scores = self.model.predict(X_test).ravel()
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        prec_rec_auc = average_precision_score(y_test, y_scores)
        plot_test_recall(precision, recall, prec_rec_auc, model_name)
    
    def predict(self, x_test, th):
        y_pred_prob = self.model.predict(x_test)
        y_pred = (y_pred_prob > th).astype(int)
        return y_pred

##############
### Models ###
##############

class BinClassifModelV1(BinaryClassifier):

    def __init__(self, n_input):
        #model
        self.model = Sequential()
        #input
        self.model.add(Dense(32, input_dim = n_input, kernel_initializer='uniform',activation='softplus'))
        #hidden layers
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(64, kernel_initializer='he_uniform', activation='softplus'))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(128, kernel_initializer='he_uniform', activation='softplus'))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(256, kernel_initializer='he_uniform', activation='softplus'))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(128, kernel_initializer='he_uniform', activation='softplus'))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(64, kernel_initializer='he_uniform', activation='softplus'))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(32, kernel_initializer='he_uniform', activation='softplus'))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(1, kernel_initializer='he_uniform', activation='sigmoid'))
        #compile
        self.model.compile(optimizer=adagrad(lr=0.05), loss='binary_crossentropy')

        #title
        self.title = 'optimizer: adagrad , lr = 0.05, loss = binary crossentropy'

        #training
        self.history = None

        #name
        self.model_name = "BCM1"

class AutoencoderModelV1(Autoencoder):

    def __init__(self, n_features, anomaly_class):

        #input
        input_layer = Input(shape=(n_features, ))

        #encode
        encoder = Dense(8,kernel_initializer='he_uniform',activation='relu')(input_layer)
        drop = Dropout(rate=0.2)(encoder)

        #latent
        latent = Dense(2, kernel_initializer='he_uniform',activation='relu')(drop)
        drop = Dropout(rate=0.2)(latent)

        #decode
        decoder = Dense(8, kernel_initializer='he_uniform',activation='relu')(drop)
        drop = Dropout(rate=0.2)(decoder)

        #output
        output_layer = Dense(n_features, activation="sigmoid")(drop)
        self.model = Model(inputs=input_layer, outputs=output_layer)

        #compile
        self.model.compile(loss='mean_squared_error', optimizer=adam(lr=0.01))

        #title
        self.title = 'optimizer: adam , lr = 0.01, loss = mean squared error'

        #training
        self.history = None

        #name
        self.model_name = "AM1"

        #anomaly class
        self.anomaly_class = anomaly_class