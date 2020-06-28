import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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