import pandas as pd
import json
import numpy as np
import math

#abstract
from abc import ABC, abstractmethod

#keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import adagrad, adam, adamax
from keras.models import Model
from keras.models import model_from_json

#sklearn
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, roc_auc_score,
precision_recall_curve, average_precision_score)
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

#plot
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from ada.plot import plot_train_loss, plot_test_roc, plot_test_recall



class KerasModel(ABC):
    """Mother class for keras models"""
    def __init__(self, n_input):
        self.model = None
        self.history = None
        self.title = ""
        self.model_name = ""
    
    def fit(self, X_train, y_train, w_train, X_val, y_val, w_val, epochs, class_weights = None, verbose = 1):
        if self.model is not None:
            if class_weights != None:
                self.history = pd.DataFrame(self.model.fit(
                    X_train.values, y_train, sample_weight = w_train,
                    epochs = epochs,
                    verbose = verbose,
                    validation_data = (X_val.values, y_val, w_val),
                    class_weight = class_weights,
                ).history)
            else:
                self.history = pd.DataFrame(self.model.fit(
                    X_train.values, y_train, sample_weight = w_train,
                    epochs = epochs,
                    verbose = verbose,
                    validation_data = (X_val.values, y_val, w_val),
                ).history)
        else:
            print("Build your model first!")
    
    def plot_loss(self, width = 10, height = 6):
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        plot_train_loss(loss, val_loss, len(loss), width = 10, height = 6)
    
    def save(self, directory, version):
        if self.model is not None:
            #save model
            model_json = self.model.to_json()
            with open(f"{directory}/{self.model_name}_{version}.json", "w") as json_file:
                json_file.write(model_json)
            #save weights
            self.model.save_weights(f"{directory}/{self.model_name}_{version}.h5")
        if self.history is not None:
            #save training data
            with open(f"{directory}/{self.model_name}_{version}.csv", mode='w') as f:
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
    
    #added on 29 Jun 2020
    def f1(self, x_test, y_test, w_test, th):
        #prediction
        y_pred = self.predict(x_test, th)
        #f1
        f1_by_class = f1_score(y_test, y_pred, sample_weight = w_test, average = None)
        f1_wavg = f1_score(y_test, y_pred, sample_weight = w_test, average='weighted')
        return f1_by_class.tolist() + [f1_wavg, ]
    
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

    def fit(self, x_train, w_train, x_val, w_val, epochs, verbose = 1):
        super().fit(x_train, x_train, w_train, x_val, x_val, w_val, epochs, verbose = verbose)
    
    def plot_reconstruction_error(self, x_test, y_test, th):
        x_pred = self.model.predict(x_test)
        errors = np.mean(np.power(x_test.values - x_pred, 2), axis=1)
        color = ListedColormap(["#1f85ad", "#ff7b00"])
        plt.figure(figsize=(10, 6))
        plt.scatter(x = range(len(errors)), y = errors, c = y_test.flatten(), s=16, cmap = color)
        plt.axhline(y=th, color='red', linestyle='-')
        plt.show()
    
    def predict(self, x_test, th):
        x_pred = self.model.predict(x_test)
        errors = np.mean(np.power(x_test.values - x_pred, 2), axis=1)
        
        if self.anomaly_class == 1:
            y_pred = (errors > th).astype(int)
        else:
            y_pred = (errors <= th).astype(int)
            
        return y_pred
    
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
    
    def plot_errors(self, x_test):
        x_pred = self.model.predict(x_test)
        errors = np.mean(np.power(x_test.values - x_pred, 2), axis=1)
        plt.figure(figsize=(10, 6))
        plt.scatter(x = range(len(errors)), y = errors, s = 16, cmap = "#1f85ad")
        plt.show()
    
    def error_wavg_std(self, x_test, w_test):
        x_pred = self.model.predict(x_test)
        errors = np.mean(np.power(x_test.values - x_pred, 2), axis=1)

        wavg = np.average(errors, weights = w_test)
        variance = np.average((errors-wavg)**2, weights = w_test)

        return (wavg, math.sqrt(variance))
    


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

class BinaryClassifierModel1(BinaryClassifier):

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

        #training
        self.history = None

        #name
        self.model_name = "BC1"

class AutoencoderModel1(Autoencoder):

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

        #training
        self.history = None

        #name
        self.model_name = "A1"

        #anomaly class
        self.anomaly_class = anomaly_class

class AutoencoderModelForTuning(Autoencoder):

    def __init__(self, n_features, anomaly_class, optimizer, lr):

        #input
        input_layer = Input(shape = (n_features, ))

        #encode
        encoder = Dense(8,kernel_initializer = 'he_uniform',activation = 'relu')(input_layer)
        drop = Dropout(rate=0.2)(encoder)

        #latent
        latent = Dense(4, kernel_initializer = 'he_uniform',activation = 'relu')(drop)
        drop = Dropout(rate=0.2)(latent)

        #decode
        decoder = Dense(8, kernel_initializer = 'he_uniform',activation = 'relu')(drop)
        drop = Dropout(rate=0.2)(decoder)

        #output
        output_layer = Dense(n_features, activation = "sigmoid")(drop)
        self.model = Model(inputs = input_layer, outputs = output_layer)

        #compile
        self.model.compile(loss='mean_squared_error', optimizer = optimizer(lr))

        #training
        self.history = None

        #name
        self.model_name = "AT"

        #anomaly class
        self.anomaly_class = anomaly_class

class AutoencoderModel2(Autoencoder):

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
        self.model.compile(loss='mean_squared_error', optimizer=adamax(lr=0.1))

        #training
        self.history = None

        #name
        self.model_name = "A2"

        #anomaly class
        self.anomaly_class = anomaly_class

# Created on SPOOKY month 2020
class BinaryClassifierModel4(BinaryClassifier):
    def __init__(self, n_features):
        # Input
        input_layer = Input(shape=(n_features, ))

        # Hidden layers
        layer = Dense(32, kernel_initializer='uniform',activation="relu")(input_layer)
        drop = Dropout(rate=0.2)(layer)
        layer = Dense(64, kernel_initializer='he_uniform',activation="relu")(input_layer)
        drop = Dropout(rate=0.2)(layer)
        layer = Dense(128, kernel_initializer='he_uniform',activation="relu")(input_layer)
        drop = Dropout(rate=0.2)(layer)
        layer = Dense(256, kernel_initializer='he_uniform',activation="relu")(input_layer)
        drop = Dropout(rate=0.2)(layer)
        layer = Dense(128, kernel_initializer='he_uniform',activation="relu")(input_layer)
        drop = Dropout(rate=0.2)(layer)
        layer = Dense(64, kernel_initializer='he_uniform',activation="relu")(input_layer)
        drop = Dropout(rate=0.2)(layer)
        layer = Dense(32, kernel_initializer='he_uniform',activation="relu")(input_layer)
        drop = Dropout(rate=0.2)(layer)
        
        # Output
        output_layer = Dense(1, kernel_initializer="he_uniform", activation='sigmoid')(drop)

        # Model
        self.model = Model(inputs = input_layer, outputs = output_layer)

        # Compile
        self.model.compile(optimizer = adamax(0.05), loss='binary_crossentropy')

        # Training
        self.history = None

        # Name
        self.model_name = "BC4"