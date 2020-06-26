import numpy as np
from sklearn.metrics import roc_auc_score


def onehot(arr: np.ndarray, n_cls: int):
    arr_out = np.zeros((arr.size, n_cls))
    arr_out[np.arange(arr.size), arr] = 1
    return arr_out


def multiclass_roc_score(label: list, pred: list, n_cls: int):
    assert len(label) == len(pred)
    # assert max(label) <= n_cls
    # assert max(pred) <= n_cls

    arr_label = np.array(label)
    arr_pred = np.array(pred)

    onehot_label = onehot(arr_label, n_cls)
    onehot_pred = onehot(arr_pred, n_cls)

    roc = roc_auc_score(y_true=onehot_label, y_score=onehot_pred, multi_class='ovr')
    return roc
