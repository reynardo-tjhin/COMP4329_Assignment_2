import numpy as np

from sklearn.metrics import (
    accuracy_score,
    multilabel_confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

def get_precision_score(y_true: np.ndarray, y_pred: np.ndarray, display: bool = True):
    """
    Get the 4 different precision scores (micro, macro, weighted, samples).

    Arguments:
        y_true (np.ndarray | Iterables): the true labels.
        y_pred (np.ndarray | Iterables): the predicted labels.
        display (bool): whether to print the resultant values.

    Returns:
        None if display
        Otherwise, (micro, macro, weighted, samples) precision score.
    """
    if (display):
        print("Precision Score (Micro):    {:.1f}%".format(precision_score(y_true, y_pred, average='micro') * 100))
        print("Precision Score (Macro):    {:.1f}%".format(precision_score(y_true, y_pred, average='macro') * 100))
        print("Precision Score (Weighted): {:.1f}%".format(precision_score(y_true, y_pred, average='weighted') * 100))
        print("Precision Score (Sample):   {:.1f}%".format(precision_score(y_true, y_pred, average='samples') * 100))

    return (
        precision_score(y_true, y_pred, average='micro'),
        precision_score(y_true, y_pred, average='macro'),
        precision_score(y_true, y_pred, average='weighted'),
        precision_score(y_true, y_pred, average='samples')
    )

def get_recall_score(y_true: np.ndarray, y_pred: np.ndarray, display: bool = True):
    """
    Get the 4 different recall scores (micro, macro, weighted, samples).

    Arguments:
        y_true (np.ndarray | Iterables): the true labels.
        y_pred (np.ndarray | Iterables): the predicted labels.
        display (bool): whether to print the resultant values.

    Returns:
        None if display
        Otherwise, (micro, macro, weighted, samples) precision score.
    """
    if (display):
        print("Recall Score (Micro):    {:.1f}%".format(recall_score(y_true, y_pred, average='micro') * 100))
        print("Recall Score (Macro):    {:.1f}%".format(recall_score(y_true, y_pred, average='macro') * 100))
        print("Recall Score (Weighted): {:.1f}%".format(recall_score(y_true, y_pred, average='weighted') * 100))
        print("Recall Score (Sample):   {:.1f}%".format(recall_score(y_true, y_pred, average='samples') * 100))

    return (
        recall_score(y_true, y_pred, average='micro'),
        recall_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='weighted'),
        recall_score(y_true, y_pred, average='samples')
    )

def get_f1_score(y_true: np.ndarray, y_pred: np.ndarray, display: bool = True):
    """
    Get the 4 different precision scores (micro, macro, weighted, samples).

    Arguments:
        y_true (np.ndarray | Iterables): the true labels.
        y_pred (np.ndarray | Iterables): the predicted labels.
        display (bool): whether to print the resultant values.

    Returns:
        None if display
        Otherwise, (micro, macro, weighted, samples) precision score.
    """
    if (display):
        print("F1 Score (Micro):    {:.1f}%".format(f1_score(y_true, y_pred, average='micro') * 100))
        print("F1 Score (Macro):    {:.1f}%".format(f1_score(y_true, y_pred, average='macro') * 100))
        print("F1 Score (Weighted): {:.1f}%".format(f1_score(y_true, y_pred, average='weighted') * 100))
        print("F1 Score (Sample):   {:.1f}%".format(f1_score(y_true, y_pred, average='samples') * 100))

    return (
        f1_score(y_true, y_pred, average='micro'),
        f1_score(y_true, y_pred, average='macro'),
        f1_score(y_true, y_pred, average='weighted'),
        f1_score(y_true, y_pred, average='samples')
    )


# reference: https://www.kaggle.com/code/kmkarakaya/multi-label-model-evaluation
def _test():
    y_true = np.array([
        [0, 1, 1, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
    ])
    y_scores = np.array([
        [0.2, 0.6, 0.1, 0.8],
        [0.4, 0.9, 0.8, 0.6],
        [0.8, 0.4, 0.5, 0.7],
    ])

    # getting the output
    threshold = 0.5
    y_pred = np.where(y_scores >= threshold, 1, 0)

    print("Metrics:")

    # classification report
    # label_names = ["label A", "label B", "label C", "label D"]
    # can return in dictionary form:
    # classification_report(y_true, y_pred, target_names=label_names, output_dict=True)

    print("Precision Score (Micro):    {:.1f}%".format(precision_score(y_true, y_pred, average='micro') * 100))
    print("Precision Score (Macro):    {:.1f}%".format(precision_score(y_true, y_pred, average='macro') * 100))
    print("Precision Score (Weighted): {:.1f}%".format(precision_score(y_true, y_pred, average='weighted') * 100))
    print("Precision Score (Sample):   {:.1f}%".format(precision_score(y_true, y_pred, average='samples') * 100))

    print()
    print("Recall Score (Micro):    {:.1f}%".format(recall_score(y_true, y_pred, average='micro') * 100))
    print("Recall Score (Macro):    {:.1f}%".format(recall_score(y_true, y_pred, average='macro') * 100))
    print("Recall Score (Weighted): {:.1f}%".format(recall_score(y_true, y_pred, average='weighted') * 100))
    print("Recall Score (Sample):   {:.1f}%".format(recall_score(y_true, y_pred, average='samples') * 100))

    print()
    print("F1 Score (Micro):    {:.1f}%".format(f1_score(y_true, y_pred, average='micro') * 100))
    print("F1 Score (Macro):    {:.1f}%".format(f1_score(y_true, y_pred, average='macro') * 100))
    print("F1 Score (Weighted): {:.1f}%".format(f1_score(y_true, y_pred, average='weighted') * 100))
    print("F1 Score (Sample):   {:.1f}%".format(f1_score(y_true, y_pred, average='samples') * 100))


# testing
if (__name__ == "__main__"):
    _test()
