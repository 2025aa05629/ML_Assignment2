from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import numpy as np

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    try:
        y_prob = model.predict_proba(X_test)
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)

        results["AUC"] = roc_auc_score(
            y_test_bin, y_prob,
            average="weighted",
            multi_class="ovr"
        )
    except:
        results["AUC"] = "Not Supported"

    return results
