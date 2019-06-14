import numpy as np

def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))
def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == -1))
def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == -1) & (y_pred == 1))
def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == -1) & (y_pred == 0))

def build_confusion_matrix(tp, fn, fp, tn):
    return np.array([[tn,fp],[fn,tp]])


def confusion_matrix(y_true, predictions):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for index, prediction in enumerate(predictions):
        if int(prediction) == 1 and int(y_true[index]) == 1:
            tp = tp + 1
        elif int(prediction) == -1 and int(y_true[index]) == 1:
            fn = fn + 1
        elif int(prediction) == 1 and int(y_true[index]) == -1:
            fp = fp + 1
        elif int(prediction) == -1 and int(y_true[index]) == -1:
            tn = tn + 1

    return (tp, tn, fp, fn)


