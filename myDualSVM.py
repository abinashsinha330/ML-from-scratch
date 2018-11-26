import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import sys
import matplotlib.pyplot as plt
import utility as util
from abc import ABCMeta, abstractmethod


class SVM(object):
    """Abstract class having the methods that are common to all concrete SVM algorithm implementations"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def model(self): pass

    @staticmethod
    def predict(params_w, params_b, x, y):
        prediction = np.dot(params_w[:, None].T, x.T) + params_b
        prediction = np.squeeze(prediction)
        predict = np.zeros_like(y)
        for j in range(prediction.shape[0]):
            if prediction[j] > 0:
                predict[j] = 1
            else:
                predict[j] = -1
        return predict


class DSVM(SVM):
    """This class is used to use the concept of linear Support Vector Machines where we try to maximise the minimum
    margin that classifies our data into two classes. In our case, we have considered non-separable case where we
    introduce slack variables. Our function that we want to minimise becomes a Lagrange dual quadratic optimization
    problem with inequality and equality constraints using. We get this dual using Lagrange multipliers technique
    We make use of convex optimization solvers to solve the same"""

    def __init__(self, c, training_x, training_y, testing_x, testing_y):
        self.c = c
        # self.delete_useless_features()
        self.training_x = training_x
        self.training_y = training_y
        self.testing_x = testing_x
        self.testing_y = testing_y
        super(DSVM, self).__init__()
        # self.data.loc[:, 1:] = self.normalize()

    @staticmethod
    def find_parameters(a, x, y):
        """In this method we find the values of w and b by saying that alpha values which are greater than 1e-6
        correspond to our support vectors since the values smaller this threshold seem negligible for finding out
        support vectors. Getting our support vectors we then find out samples which correspond to y=-1 and y=1
        support vectors.
        We find number of both these categories of samples. We use whichever has lower value as our count value
        and pick count number of samples from each of these two sets and we calculate b using the equations
        wT.x + b = -1 and wT.x + b = 1 and average over all count number of calculations"""
        sv_indices = np.where(np.any(a > 1e-6, axis=1))
        sv_y = y[sv_indices]
        sv_x = x[sv_indices]
        sv_a = a[sv_indices]
        w = np.expand_dims(np.sum(sv_a * sv_y * sv_x, axis=0), axis=1)
        sv_minus_1_indices = np.where(np.any(sv_y == -1, axis=1))
        sv_plus_1_indices = np.where(np.any(sv_y == 1, axis=1))
        sv_x_minus_1 = sv_x[sv_minus_1_indices]
        sv_x_plus_1 = sv_x[sv_plus_1_indices]
        count = len(sv_x_minus_1) if len(sv_x_minus_1) < len(sv_x_plus_1) \
            else len(sv_x_plus_1)
        sv_x_plus_1_count_ind = np.random.randint(0, high=len(sv_x_plus_1), size=count)
        sv_x_minus_1_count_ind = np.random.randint(0, high=len(sv_x_minus_1), size=count)
        b_actual = - np.sum(np.dot((sv_x_plus_1[sv_x_plus_1_count_ind]
                                    + sv_x_minus_1[sv_x_minus_1_count_ind]), w)) / float(count)
        return w, b_actual, len(sv_y)

    def model(self):
        a = self.solve(self.training_x, self.training_y)
        param_w, param_b, num_support_vectors = DSVM.find_parameters(a, self.training_x, self.training_y)
        test_predict = self.predict(param_w, param_b, self.testing_x, self.testing_y)
        training_predict = self.predict(param_w, param_b, self.training_x, self.training_y)

        return param_w, param_b, test_predict, training_predict, num_support_vectors

    def solve(self, x, y):
        """This method is used to solve the Lagrangian dual problem of the non-separable case of SVM using the convex
        optimizer module"""
        num_examples = x.shape[0]
        m = y * x
        p = matrix(np.dot(m, m.T).astype(float))
        q = matrix(-np.ones((num_examples, 1)))
        g1 = -1 * np.eye(num_examples)
        g2 = 1 * np.eye(num_examples)
        h1 = np.zeros(num_examples)
        h2 = np.ones(num_examples) * self.c

        g = matrix(np.vstack((g1, g2)))
        h = matrix(np.hstack((h1, h2)))

        A = matrix(y.T.astype(float))
        b = matrix(np.zeros(1))
        solvers.options['show_progress'] = False
        sol = solvers.qp(p, q, g, h, A, b)
        a = np.array(sol['x'])

        return a


if __name__ == "__main__":

    filename = sys.argv[1]
    count = 0
    c = [0.01, 0.1, 1, 10, 100]
    for arg in sys.argv:
        if count > 1:
            c.append(float(arg))
    data = pd.read_csv(filename, header=None, index_col=False)
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == 3:
            data.iloc[i, 0] = -1
    data = util.delete_useless_features(data)
    data.loc[:, 1:] = util.normalize(data)
    # we test our SVM implements using the MNIST dataset which has 2000 training examples here with 28*28 = 784 features
    # first argument is name of the file, second argument is C value
    mean_test_errors = []
    mean_training_errors = []
    mean_geo_margins = []
    mean_num_svs = []
    print('\nSVM using Lagrange dual and convex optimization solver')
    for l in c:
        test_error_fractions = []
        training_error_fractions = []
        geo_margins = []
        num_svs = []
        for _ in range(10):
            # perform the 80-20 random-specific split
            training_data, test_data = util.random_class_specific_split(data)
            train_x = np.array(training_data.loc[:, 1:])
            train_y = np.expand_dims(np.array(training_data.loc[:, 0]), axis=1)
            test_x = np.array(test_data.loc[:, 1:])
            test_y = np.expand_dims(np.array(test_data.loc[:, 0]), axis=1)
            dsvm = DSVM(l, train_x, train_y, test_x, test_y)
            param_w, b, test_predicts, training_predicts, num_sv = dsvm.model()
            test_error = util.calculate_error(test_predicts, test_y)
            training_error = util.calculate_error(training_predicts, train_y)
            test_error_fractions.append(test_error)
            training_error_fractions.append(training_error)
            geo_margins.append(1 / np.linalg.norm(param_w))
            sv = (np.dot(train_x, param_w) + b) * train_y
            num_svs.append(num_sv)
        mean_test_error = np.mean(test_error_fractions) * 100
        mean_test_errors.append(mean_test_error)
        mean_training_error = np.mean(training_error_fractions) * 100
        mean_training_errors.append(mean_training_error)
        stddev_test_error = np.std(test_error_fractions) * 100
        stddev_training_error = np.std(training_error_fractions) * 100
        mean_geo_margin = np.mean(geo_margins)
        mean_geo_margins.append(mean_geo_margin)
        mean_num_sv = np.mean(num_svs)
        mean_num_svs.append(mean_num_sv)
        print('\nFor c value of ', l)
        print('\nMean of test errors in %: ', mean_test_error)
        print('\nMean of training errors in %', mean_training_error)
        print('\nStandard deviation in test errors in %: ', stddev_test_error)
        print('\nStandard deviation in training errors in %: ', stddev_training_error)
        print('\nMean geometric margin: ', mean_geo_margin)
        print('\nMean number of support vectors: ', mean_num_sv)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].set_title('Geometric Margin vs. C')
    axes[0, 0].plot(c, mean_geo_margins)
    axes[0, 1].set_title('Number of Support Vectors vs. C')
    axes[0, 1].plot(c, mean_num_svs)
    axes[1, 0].set_title('Test Error Percentage vs. C')
    axes[1, 0].plot(c, mean_test_errors)
    axes[1, 1].set_title('Training Error Percentage vs. C')
    axes[1, 1].plot(c, mean_training_errors)
    plt.tight_layout()
    plt.savefig('dual_plots.pdf', transparent=True, dpi=600)
