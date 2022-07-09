import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support
import itertools


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    #print(preds_flat)
    #print(len(preds_flat))
    #print(labels_flat)
    #print(len(labels_flat))
    return f1_score(labels_flat, preds_flat, average='weighted')
    print(f'F1 score: {F1_score}')

def precision_recall_fscore_support_func(true_vals, predictions):
    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_vals.flatten()
    return precision_recall_fscore_support(labels_flat, preds_flat, average='weighted'), precision_recall_fscore_support(labels_flat, preds_flat, average='micro'), precision_recall_fscore_support(labels_flat, preds_flat, average='macro')

def classification_report_func(true_vals, predictions):
    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_vals.flatten()    
    return classification_report(labels_flat, preds_flat)


def cnf_matrix_func(true_vals, predictions,label_dict):
    target_names = []
    label_dict_reversed = {}
    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_vals.flatten()     

    for key, value in label_dict.items():
        label_dict_reversed[value] = key
        
    list_of_values=sorted(set(labels_flat)) 
    
    #use the original label names(=main, background, none) instead of 0, 1, 2
    for x in list_of_values:        
        if x in label_dict_reversed:
            target_names.append(label_dict_reversed[x])

    cnf_matrix = confusion_matrix(labels_flat, preds_flat)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(10,9))
    plot_confusion_matrix(cnf_matrix, classes=target_names,
    title='Confusion matrix')
    plt.show()


# we make an inversed dictionary, instead of value to key, we take key to value
#F score better than accuracy in a situation where we have imbalanced data



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):


        if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
        else:
                print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", fontsize=14, color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('Predicted label')
        plt.xlabel('True label')
        plt.tight_layout()

