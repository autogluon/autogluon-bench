def f1_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    TP = ((y_true == 1) & (y_pred == 1)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0

    return 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
