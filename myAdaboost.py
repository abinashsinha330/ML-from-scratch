import pandas as pd
import numpy as np
import sys
import utility as util
from myDecisionTree import DecisionStumpClassifier
from abc import ABCMeta, abstractmethod
import copy


class Boosting(object):
    """Abstract class having the methods that are common to all concrete boosting algorithm implementations"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def model(self, *args): pass

    @staticmethod
    def predict(predicts, alpha, weak_predicts):
        return predicts + alpha * weak_predicts


class AdaBoostClassifier(Boosting):
    """Implements Adaboost with the chosen weak classifier"""

    def __init__(self, seq_features):
        super(AdaBoostClassifier, self).__init__()
        self.seq_features = seq_features

    def model(self, num_rounds, x, y, decision_stumps):
        """core method implementing the logic of Adaboost algorithm"""

        weak_classifiers_list = []
        alpha_values = []
        epsilons = []
        n = len(y)
        predicts = np.zeros_like(y)
        # Initialize each weight as 1/n
        w = np.ones_like(y) / n
        d_stumps_copy = copy.deepcopy(decision_stumps)
        for t in range(num_rounds):
            print("\nSelecting decision stump with least weighted error")
            best_d_stump, weak_predicts_t = DecisionStumpClassifier.model(d_stumps_copy, x, y, w)
            best_feature = best_d_stump.feature_index
            print("\nBest decision stump uses {} feature".format(best_feature))
            # indicator function value
            indicator_values_t = np.zeros_like(y)
            for i in range(n):
                if weak_predicts_t[i] != y[i]:
                    indicator_values_t[i] = 1
            # epsilon error
            sum_w = np.sum(w)
            epsilon_t = w.T.dot(indicator_values_t) / sum_w
            # alpha
            alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
            # normalization factor
            z_t = np.sum(w * np.exp(-alpha_t * y * weak_predicts_t))
            # new weights
            w = (w * np.exp(-alpha_t * y * weak_predicts_t)) / z_t
            # add weighted predictions while iterating itself
            predicts = AdaBoostClassifier.predict(predicts, alpha_t, weak_predicts_t)
            alpha_values.append(alpha_t)
            weak_classifiers_list.append(best_d_stump)
            epsilons.append(epsilon_t)
            # don't use the stump in next iteration so that we pick the next best weak learner in next iteration
            d_stumps_copy.remove(best_d_stump)

        predicts = np.sign(predicts)
        return predicts, alpha_values, weak_classifiers_list, epsilons


if __name__ == '__main__':

    filename = 'dog_adoption.csv'  # sys.argv[1]
    num_runs = 4  # int(sys.argv[2])
    data = pd.read_csv(filename, header=None, index_col=False)
    data_x, data_y = np.array(data.iloc[1:, 1:-1], dtype=float), np.array(data.iloc[1:, -1], dtype=float).reshape(-1, 1)
    num_features = data_x.shape[1]
    sequence_features = list()
    # Make the decision stumps using one feature at a time
    sequence_features.append(0)
    sequence_features.append(1)
    sequence_features.append(2)
    sequence_features.append(3)
    sequence_features.append(0)
    sequence_features.append(1)
    sequence_features.append(2)
    sequence_features.append(3)
    sequence_features.append(0)
    sequence_features.append(1)
    print("Running Adaboost algorithm")
    # set of thresholds pre-decided
    feature_wise_thresh = dict()
    feature_wise_thresh[0] = 3  # if porty preparedness level >= 3 then normal otherwise low and there are 1/12 errors
    feature_wise_thresh[1] = 33.33  # if price >= 33.33 then high otherwise low and there are 4/12 errors
    feature_wise_thresh[2] = 2  # if # of carpet damage instances >=2 then high otherwise low and there are 2/12 errors
    feature_wise_thresh[3] = 1  # color BROWN/WHITE = 1 and YELLOW = 0 and there are 4/12 errors

    # build the 4 decisions stumps using the 4 features of the data
    d_stumps = list()
    for feature_index in range(num_features):
        print("\nBuilding decision stump for feature: ", data.iloc[0, feature_index+1])
        d_stump = DecisionStumpClassifier(feature_index, feature_wise_thresh[feature_index])
        d_stump.build_stump(data_x, data_y)
        d_stumps.append(d_stump)
    # Fit Adaboost classifier using a decision stumps using each feature as
    # weak classifier picking all of them during all iterations
    adaboost = AdaBoostClassifier(sequence_features)
    predictions, alphas, weak_classifiers, epsilons = adaboost.model(num_runs, data_x, data_y, d_stumps)
    error_fraction = util.calculate_error(predictions, data_y)

    print("\nFinal error percentage: ", error_fraction*100)
    print("\nWeights of decision stumps: feature used and weight of its decision stump: ", alphas)
    print("\nSequence of best features for decision stumps selected in Adaboost: ",
          [data.iloc[0, x.feature_index+1] for x in weak_classifiers])
    print("\nEpsilon error fractions while running Adaboost ", epsilons)

    test_joey_porty = float(sys.argv[3])
    test_joey_cost = float(sys.argv[4])
    test_joey_carpet = float(sys.argv[5])
    test_joey_color = float(sys.argv[6])
    test_joey = np.array([[test_joey_porty, test_joey_cost, test_joey_carpet, test_joey_color]])
    predict_joey = np.zeros((1, 1))
    for weak_learner, alpha in zip(weak_classifiers, alphas):
        predict_joey = predict_joey + (alpha * DecisionStumpClassifier.predict(weak_learner, test_joey, predict_joey))

    predict_joey = np.sign(predict_joey)
    print("\n{}! According to my model, you {} Joey".format('Congrats' if predict_joey[0, 0] == 1 else 'Sorry',
                                                            'SHOULD ADOPT' if predict_joey[0, 0] == 1
                                                            else 'SHOULDN\'T ADOPT'))


