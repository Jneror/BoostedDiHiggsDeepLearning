import ROOT
import pandas as pd
from root_numpy import tree2array
from os import path
from glob import glob
import matplotlib.pyplot as plt
#dnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, roc_auc_score,
precision_recall_curve, average_precision_score)
from keras.models import model_from_json

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import adagrad
#abstract classes
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
    """Binary classifying of events"""
    df[class_column] = (df["sample"].str.contains(f".*{signal}[^0-9]*",regex = True)).astype(int)
    return df

def preprocess_df(df, signal, class_col = "label", region = None, tag = None):
    """Preprocess: drop two dim, filter region if any, filter tag if any and classify signals"""
    #drop twodim region, because we are not going to use it
    df = drop_twodim(df)
    if region is not None: df = filter_region(df, region)
    if tag is not None: df = filter_tag(df, tag)
    return classify_events(df, signal, class_col)

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

def pop_col_from_dfs(dfs, col):
    """Drop the column from a list of dfs and return this columns in form of array"""
    popped_cols = [df.pop(col).values for df in dfs]
    return popped_cols

def standar_scale_dfs(dfs):
    scaler = StandardScaler().fit(dfs[0])
    return [pd.DataFrame(scaler.transform(df),columns = df.columns) for df in dfs]

def rotate_vectors(vectors):
    return [v.reshape(-1, 1) for v in vectors]

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

def pop_col(X, col):
    """Returns the popped column with the new dataframe"""
    #pandas has a pop column but modifies the original dataframe, which i dont want
    return X[col].values, X.drop(columns = [col])

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

#get data v1
def get_trainvaltest_from_dataset(data_path, signal, region = None, tag = None,
    train_size = 0.6, val_size = 0.2, test_size = 0.2, seed = 420):

    if region not in {"SR", "QCDCR", None}:
        print("Error: Region not valid!")
        return
    
    if tag not in {0, 1, 2, None}:
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

    #drop two dim
    data = drop_twodim(data)

    #filter region and tag
    if region is not None:
        data = filter_region(data, region)

    if tag is not None:
        data = filter_tag(data, tag)

    #classify (label)
    data = classify_events(data, signal, "label")

    data = data[["EventWeight", "label"] + selected_features]

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

#get data v2
def get_trainvaltest_from_csv(data_path, signal, train_size, val_size, test_size, seed, region = None, tag = None):
    #check errors
    if region not in {"SR", "QCDCR", None}: print("Error: Region not valid!")
    if tag not in {0, 1, 2, None}: print("Error: Tag not valid")
    if train_size + val_size + test_size != 1.0: print("Error: Datasets sizes dont add 1")
    if not path.exists(f"{data_path}/{signal}.csv"): print("Error: Dataset not found")

    #read csv
    df = pd.read_csv(f"{data_path}/{signal}.csv")

    #preprocess
    df = preprocess_df(df, signal, region = region, tag = tag, class_col = "label")
    
    #columns to pop from the X set later
    cols_to_pop = [col_names["weight"], "label"]
    if tag is None: cols_to_pop.append(col_names["tag"])
    if region is None: cols_to_pop.append(col_names["region"])

    #leave the selected features and the cols to pop later
    df = df[cols_to_pop + selected_features]

    #split into x and y
    x = df.drop(columns = ["label"])
    y = df["label"].values

    #object where all the datasets will be stored
    datasets = {}

    #split into train, val and test datasets
    x_train, x_val, x_test, y_train, y_val, y_test = trainvaltest_split(x, y, seed, 0.6, 0.2, 0.2)

    #all the datasets for train, val and test will be stored here
    w_train, w_val, w_test = pop_col_from_dfs([x_train, x_val, x_test], col_names["weight"])
    datasets["w"] = {"train": w_train, "val": w_val, "test": w_test}

    #pop filter and tag column if there are no filters
    if tag is None:
        tags = pop_col_from_dfs([x_train, x_val, x_test], col_names["tag"])
        datasets["tag"] = {"train": tags[0], "val": tags[1], "test": tags[2]}

    if region is None:
        regions = pop_col_from_dfs([x_train, x_val, x_test], col_names["region"])
        datasets["region"] = {"train": regions[0], "val": regions[1], "test": regions[2]}

    #scale
    x_train, x_val, x_test = standar_scale_dfs([x_train, x_val, x_test])
    datasets["x"] = {"train": x_train, "val": x_val, "test": x_test}

    #reshape y
    y_train, y_val, y_test = rotate_vectors([y_train, y_val, y_test])
    datasets["y"] = {"train": y_train, "val": y_val, "test": y_test}

    return datasets


#################
### Models v1 ###
#################

class DeepNeuralNetworkModel(ABC):
    """Abstract mother of models"""

    @abstractmethod
    def fit(self, X_train, y_train, w_train, X_val, y_val, w_val, epochs):
        print("Implement!")
    
    @abstractmethod
    def plot_training(self, width = 10, height = 6):
        print("Implement!")

    @abstractmethod
    def save(self, directory, version):
        print("Implement!")
    
    @abstractmethod
    def load(self, directory, version):
        print("Implement!")
    
    @abstractmethod
    def evaluate(self, X_test, y_test, w_test):
        print("Implement!")


class KerasModelGamma(DeepNeuralNetworkModel):
    """First stable version of the mother class for keras models"""
    def __init__(self, n_input):
        self.model = None
        self.history = None
        self.title = ""
        self.model_name = ""
    
    def fit(self, X_train, y_train, w_train, X_val, y_val, w_val, epochs):
        if self.model is not None:
            self.history = pd.DataFrame(self.model.fit(
                X_train.values,
                y_train,
                sample_weight = w_train,
                epochs = epochs,
                verbose = 1,
                validation_data = (
                    X_val.values,
                    y_val,
                    w_val
                )
            ).history)
        else:
            print("Build your model first!")
    
    def plot_training(self, width = 10, height = 6):
        #data
        train_loss = self.history['loss']
        val_loss = self.history['val_loss']
        epochs = len(train_loss)
        #plot
        plt.figure(1, figsize=(width, height))
        plt.plot(range(epochs), train_loss)
        plt.plot(range(epochs), val_loss)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training vs Validation Loss " + self.title)
        plt.grid(True)
        plt.legend(['Training', 'Validation'])
        plt.show()
    
    def summary(self):
        self.model.summary()
    
    def save(self, directory, version):
        if self.model is not None:
            print("modelo")
            #save model
            model_json = self.model.to_json()
            with open(f"{directory}/{self.model_name}_v{version}.json", "w") as json_file:
                json_file.write(model_json)
                print("modelo")
            #save weights
            self.model.save_weights(f"{directory}/{self.model_name}_v{version}.h5")
        if self.history is not None:
            print("historia")
            #save training data
            with open(f"{directory}/{self.model_name}_v{version}.csv", mode='w') as f:
                print("mas historia")
                self.history.to_csv(f)

    
    def load(self, directory, version, model_name = None):
        if model_name is not None:
            self.model_name = model_name
        # load json and create model
        if self.model is None:
            json_file = open(f"{directory}/{self.model_name}_v{version}.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
        #load weights
        self.model.load_weights(f"{directory}/{self.model_name}_v{version}.h5")
        #load train history
        self.history = pd.read_csv(f"{directory}/{self.model_name}_v{version}.csv")
    
    def evaluate_with_weights(self, X_test, y_test, w_test, threshold = 0.4):
        y_pred_prob = self.model.predict(X_test)
        #parameter
        y_pred = (y_pred_prob > threshold)

        print("Classification Report")
        print(classification_report(y_test, y_pred, sample_weight = w_test))
        print("Confussion Matrix")
        print(confusion_matrix(y_test, y_pred, sample_weight = w_test))
    
    def evaluate(self, X_test, y_test, threshold = 0.4):
        y_pred_prob = self.model.predict(X_test)
        #parameter
        y_pred = (y_pred_prob > threshold)

        print("Classification Report")
        print(classification_report(y_test, y_pred))
        print("Confussion Matrix")
        print(confusion_matrix(y_test, y_pred))

    def plot_roc(self, X_test, y_test):
        y_scores = self.model.predict(X_test).ravel()

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = roc_auc_score(y_test, y_scores)

        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='{} (area = {:.3f})'.format(self.model_name, roc_auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

        return fpr, tpr, roc_auc
    
    def plot_recall(self, X_test, y_test):
        y_scores = self.model.predict(X_test).ravel()
        precision, recall, _ = precision_recall_curve(y_test, y_scores)

        prec_rec_auc = average_precision_score(y_test, y_scores)

        plt.figure()
        plt.plot([0, 1], [1, 0], 'k--')
        plt.plot(recall, precision, label='{} (area = {:.3f})'.format(self.model_name, prec_rec_auc))
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-recall curve')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

        return precision, recall, prec_rec_auc

#################
### Models v2 ###
#################

class ATLASModel(ABC):
    """Abstract mother of models"""

    @abstractmethod
    def fit(self, X_train, y_train, w_train, X_val, y_val, w_val, epochs):
        print("Implement!")
    
    @abstractmethod
    def plot_loss(self, width = 10, height = 6):
        print("Implement!")
    
    @abstractmethod
    def plot_acc(self, width = 10, height = 6):
        print("Implement!")

    @abstractmethod
    def save(self, directory, version):
        print("Implement!")
    
    @abstractmethod
    def load(self, directory, version):
        print("Implement!")

class KerasModel(ATLASModel):
    """Mother class for keras models"""
    def __init__(self, n_input):
        self.model = None
        self.history = None
        self.title = ""
        self.model_name = ""
    
    def fit(self, X_train, y_train, w_train, X_val, y_val, w_val, epochs):
        if self.model is not None:
            self.history = pd.DataFrame(self.model.fit(
                X_train.values,
                y_train,
                sample_weight = w_train,
                epochs = epochs,
                verbose = 1,
                validation_data = (
                    X_val.values,
                    y_val,
                    w_val
                )
            ).history)
        else:
            print("Build your model first!")
    
    def plot_loss(self, width = 10, height = 6):
        #data
        train_loss = self.history['loss']
        val_loss = self.history['val_loss']
        epochs = len(train_loss)
        #plot
        plt.figure(1, figsize=(width, height))
        plt.plot(range(epochs), train_loss)
        plt.plot(range(epochs), val_loss)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title("Training vs Validation Loss " + self.title)
        plt.grid(True)
        plt.legend(['Training', 'Validation'])
        plt.show()
    
    def plot_acc(self, width = 10, height = 6):
        if 'acc' not in self.history:
            print("No Accuracy metrics found!")
            return
        #data
        train_loss = self.history['acc']
        val_loss = self.history['val_acc']
        epochs = len(train_loss)
        #plot
        plt.figure(1, figsize=(width, height))
        plt.plot(range(epochs), train_loss)
        plt.plot(range(epochs), val_loss)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.title("Training vs Validation Accuracy " + self.title)
        plt.grid(True)
        plt.legend(['Training', 'Validation'])
        plt.show()
    
    def summary(self):
        self.model.summary()
    
    def save(self, directory, version):
        if self.model is not None:
            print("modelo")
            #save model
            model_json = self.model.to_json()
            with open(f"{directory}/{self.model_name}_v{version}.json", "w") as json_file:
                json_file.write(model_json)
                print("modelo")
            #save weights
            self.model.save_weights(f"{directory}/{self.model_name}_v{version}.h5")
        if self.history is not None:
            print("historia")
            #save training data
            with open(f"{directory}/{self.model_name}_v{version}.csv", mode='w') as f:
                print("mas historia")
                self.history.to_csv(f)

    
    def load(self, directory, version, model_name = None):
        if model_name is not None:
            self.model_name = model_name
        # load json and create model
        if self.model is None:
            json_file = open(f"{directory}/{self.model_name}_v{version}.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
        #load weights
        self.model.load_weights(f"{directory}/{self.model_name}_v{version}.h5")
        #load train history
        self.history = pd.read_csv(f"{directory}/{self.model_name}_v{version}.csv")

class Autoencoder(KerasModel):
    def __init__(self, n_input):
        super().__init__(n_input)

    def fit(self, x_train, w_train, x_val, w_val, epochs):
        super().fit(x_train, x_train, w_train, x_val, x_val, w_val, epochs)
    
    def evaluate(self, x_test):
        loss, acc = (0, 0)
        if self.model is not None:
            loss, acc = self.model.evaluate(x_test, x_test, batch_size=128)
        return loss, acc


class BinaryClassifier(KerasModel):

    def __init__(self, n_input):
        super().__init__(n_input)

    def evaluate_with_weights(self, X_test, y_test, w_test, threshold = 0.4):
        y_pred_prob = self.model.predict(X_test)
        #parameter
        y_pred = (y_pred_prob > threshold)

        print("Classification Report")
        print(classification_report(y_test, y_pred, sample_weight = w_test))
        print("Confussion Matrix")
        print(confusion_matrix(y_test, y_pred, sample_weight = w_test))
    
    def evaluate(self, X_test, y_test, threshold = 0.4):
        y_pred_prob = self.model.predict(X_test)
        #parameter
        y_pred = (y_pred_prob > threshold)

        print("Classification Report")
        print(classification_report(y_test, y_pred))
        print("Confussion Matrix")
        print(confusion_matrix(y_test, y_pred))

    def plot_roc(self, X_test, y_test):
        y_scores = self.model.predict(X_test).ravel()

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = roc_auc_score(y_test, y_scores)

        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='{} (area = {:.3f})'.format(self.model_name, roc_auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

        #return fpr, tpr, roc_auc
    
    def plot_recall(self, X_test, y_test):
        y_scores = self.model.predict(X_test).ravel()
        precision, recall, _ = precision_recall_curve(y_test, y_scores)

        prec_rec_auc = average_precision_score(y_test, y_scores)

        plt.figure()
        plt.plot([0, 1], [1, 0], 'k--')
        plt.plot(recall, precision, label='{} (area = {:.3f})'.format(self.model_name, prec_rec_auc))
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-recall curve')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

        #return precision, recall, prec_rec_auc

class BinaryClassifier1(BinaryClassifier):

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
        self.model_name = "BC1"