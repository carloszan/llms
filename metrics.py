from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def metrics(true, pred):
    average = 'weighted'
    precision = precision_score(true, pred, average=average)
    recall = recall_score(true, pred, average=average)
    f1 = f1_score(true, pred, average=average)
    accuracy = accuracy_score(true, pred)

    print(
        f"Precision: {round(precision, 3)}, Recall: {round(recall, 3)}, F1: {round(f1, 3)}, Accuracy: {round(accuracy, 3)}")
