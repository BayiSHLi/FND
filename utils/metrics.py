
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def get_confusionmatrix_fnd(preds, labels):
    # label_predicted = np.argmax(preds, axis=1)
    label_predicted = preds
    acc = accuracy_score(labels, label_predicted)
    report = classification_report(labels, label_predicted, labels=[0.0, 1.0], target_names=['real', 'fake'],digits=4)
    cm = confusion_matrix(labels, label_predicted, labels=[0,1])
    print (f"accuracy score: {acc}")
    print(report)
    print (cm)
    return acc, report, cm


def metrics(y_true, y_pred):
    metrics = {}
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['acc'] = accuracy_score(y_true, y_pred)

    return metrics
