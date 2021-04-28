import sklearn
import numpy as np

class Metrics:
    def __init__(self, predictions, actual):
        self.yhat= predictions
        self.y= actual

    def confusion_matrix(self):
        """Prints the confusion matrix of the classification
        """
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(self.y, self.yhat))

    def classification_metrics(self):
        """Returns dictionary containing classification metrics of interest
        """
        