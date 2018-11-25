import numpy as np
from abc import ABCMeta, abstractmethod


class DecisionTree(object):
    """Abstract class having the methods that are common to all concrete SVM algorithm implementations"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_stump(self, x, y): pass

    @staticmethod
    def model(*args): pass


class DecisionStumpClassifier(DecisionTree):

    def __init__(self, feature_index, thresh_val):
        super(DecisionStumpClassifier, self).__init__()
        self.feature_index = feature_index
        self.thresh_val = thresh_val
        self.branch = {}

    @staticmethod
    def model(d_stumps, x, y, w):
        best_stump = None
        best_predict = np.zeros_like(y)
        min_error = np.inf
        # numSteps = 10.0
        # for i in range(num_features):
        #     rangeMin = min(x[:, i])
        #     rangeMax = max(x[:, i])
        #     stepSize = (rangeMax - rangeMin) / numSteps
        # for j in range(-1, int(numSteps) + 1):
        # threshVal = rangeMin + float(j) * stepSize
        for d_stump in d_stumps:
            error = np.ones_like(y)
            pred_values = DecisionStumpClassifier.predict(d_stump, x, y)
            error[pred_values == y] = 0
            weighted_error = w.T.dot(error)
            if weighted_error <= min_error:
                min_error = weighted_error
                best_predict = pred_values.copy()
                best_stump = d_stump

        return best_stump, best_predict

    @staticmethod
    def predict(d_stump, x, y):
        pred_values = np.zeros_like(y)
        feature_values = x[:, d_stump.feature_index]
        thresh_val = d_stump.thresh_val
        if d_stump.branch['left'] == 1:
            pred_values[feature_values >= thresh_val] = 1
            pred_values[feature_values < thresh_val] = -1
        else:
            pred_values[feature_values >= thresh_val] = -1
            pred_values[feature_values < thresh_val] = 1
        return pred_values

    def build_stump(self, x, y):
        """The left branch indicates the values greater than or equal to threshold and right branch for smaller than
        threshold"""
        branch = self.branch
        upper_count = y[x[:, self.feature_index] >= self.thresh_val]
        sum_upper_count = np.sum(upper_count)
        lower_count = y[x[:, self.feature_index] < self.thresh_val]
        sum_lower_count = np.sum(lower_count)
        if sum_upper_count >= 0:
            branch['left'] = 1  # more +1, so above threshold prediction is +1
        if sum_upper_count < 0:
            branch['left'] = -1  # more -1, so above threshold prediction is -1
        if sum_lower_count >= 0:
            branch['right'] = 1  # more +1, so above threshold prediction is +1
        if sum_lower_count < 0:
            branch['right'] = -1  # more -1, so above threshold prediction is -1
        self.branch = branch







