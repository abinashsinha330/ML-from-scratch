import numpy as np
import pandas as pd
import math as math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys


class NaiveBayesClassifier:
    """This class if for conducting Naive Bayes classification which assumes the conditional independence between the
    probability of each feature value of input vector given the class value"""

    def __init__(self, filename, num_splits=10, train_percents=[10, 25, 50, 75, 100]):
        if not filename:
            raise ValueError('Please provide a file to retrieve data from')
        self.filename = filename
        self.data = self.preprocess(pd.read_csv(filename, header=None, index_col=False), filename)
        self.num_splits = num_splits
        self.train_percents = train_percents
        self.training_data, self.test_data = self.random_class_specific_split()
        self.classes = sorted(list(self.data.iloc[:, -1].unique()))

    @staticmethod
    def preprocess(data, filename):
        if filename == 'boston.csv':
            threshold = data.iloc[:, -1].quantile(50 / 100.0)
            print(data.shape[0])
            for i in range(data.shape[0]):
                if data.iloc[i, -1] >= threshold:
                    data.iloc[i, -1] = 1
                else:
                    data.iloc[i, -1] = 0

        return data

    @staticmethod
    def groupby_class(input_data):
        grouped_input_data = input_data.groupby(input_data.iloc[:, -1])
        return grouped_input_data

    @staticmethod
    def univariate_gaussian_pdf(x, mean, stddev):
        if stddev > 0:
            cons = (1 / (math.sqrt(2 * math.pi) * stddev))
            pdf = cons * math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stddev, 2))))
        else:
            pdf = 1
        return pdf

    @staticmethod
    def drop_lastcol(input_data):
        return input_data.drop(input_data.columns[[-1]], axis=1)

    @staticmethod
    def calculate_error(input_data, labels):
        errors = np.sum(labels != input_data.iloc[:, -1])
        return errors

    def random_class_specific_split(self):
        """Function that performs random class-specific 80-20 split i.e. 80% of data belonging to each class is taken
        together to form training data and 20% of data belonging to each class taken together to form test data"""
        grouped_data = NaiveBayesClassifier.groupby_class(self.data)
        classes = sorted(list(self.data.iloc[:, -1].unique()))
        training_data = pd.DataFrame()
        test_data = pd.DataFrame()
        for c in classes:
            class_c_data = grouped_data.get_group(c)
            test_data = pd.concat((test_data, class_c_data.sample(frac=0.20)), ignore_index=True)
            training_data = pd.concat((training_data, class_c_data.append(test_data).drop_duplicates(keep=False)),
                                      ignore_index=True)

        return training_data, test_data

    def get_percent_train_data(self, p, whole_train_data):
        """Function to give the p % of whole data which would be used for training our logistic regression"""
        grouped_data = NaiveBayesClassifier.groupby_class(whole_train_data)
        percent_training_data = pd.DataFrame()
        for c in self.classes:
            class_c_data = grouped_data.get_group(c)
            temp_data = class_c_data.sample(frac=p)
            percent_training_data = pd.concat((percent_training_data, temp_data), ignore_index=True)

        return percent_training_data

    def model(self):
        sum_errors = {}
        for percent in self.train_percents:
            sum_errors[percent] = 0
        for i in range(self.num_splits):
            whole_training_data, test_data = self.random_class_specific_split()
            for percent in self.train_percents:
                training_data = self.get_percent_train_data(percent / 100.0, whole_training_data)
                grouped_training_data = NaiveBayesClassifier.groupby_class(training_data)
                classes = sorted(list(training_data.iloc[:, -1].unique()))
                len_training_data = training_data.shape[0]
                len_test_data = test_data.shape[0]
                means = {}
                stddevs = {}
                prior_y = {}
                labels_list = []

                for c in classes:
                    class_c_data = grouped_training_data.get_group(c)
                    class_c_training_input = NaiveBayesClassifier.drop_lastcol(class_c_data)
                    means[c] = np.expand_dims(np.array(class_c_training_input.mean(axis=0)), axis=1)
                    stddevs[c] = np.expand_dims(np.array(class_c_training_input.std(axis=0)), axis=1)
                    prior_y[c] = class_c_training_input.shape[0] / float(len_training_data)

                x = NaiveBayesClassifier.drop_lastcol(test_data)
                for j in range(len_test_data):
                    label_x = 0
                    max_likelihood_x_given_c = 0
                    for c in classes:
                        # loop to calculate conditional probability of each test_input given class c and select the maximum
                        # of all probabilities for each class c to decide which class test_input will be classified into
                        posterior_x_given_c = 1
                        for k in range(x.shape[1]):
                            x_jk = x.iat[j, k]
                            posterior_xjk_given_c = NaiveBayesClassifier.univariate_gaussian_pdf(x_jk, means[c].item(k),
                                                                                                 stddevs[c].item(k))
                            posterior_x_given_c *= posterior_xjk_given_c
                        likelihood_x_given_c = prior_y[c] * posterior_x_given_c
                        if likelihood_x_given_c > max_likelihood_x_given_c:
                            max_likelihood_x_given_c = likelihood_x_given_c
                            label_x = c

                    labels_list.append(label_x)

                labels = np.array(labels_list)

                test_error = NaiveBayesClassifier.calculate_error(test_data, labels) / float(len_test_data)

                sum_errors[percent] += test_error

        mean_test_errors_map = {}
        for percent in self.train_percents:
            mean_test_errors_map[percent] = sum_errors[percent] / float(self.num_splits)
        self.plot_learning_curve(mean_test_errors_map)
        return mean_test_errors_map

    def plot_learning_curve(self, mean_test_errors_map):
        x = self.train_percents
        y = mean_test_errors_map.values()
        plt.plot(x, y)
        plt.xlabel('Percent of training set used for training')
        plt.ylabel('Mean test error over {} random splits'.format(self.num_splits))
        plt.title('Learning Curve (Gaussian Naive Bayes for {} data)'.format(self.filename))
        # function to show the plot
        # plt.show()
        plt.savefig('Learning_Curve_{}_NaiveBayes.png'.format(self.filename))


if __name__ == '__main__':
    fname = sys.argv[1]
    n_splits = sys.argv[2]
    train_spls = []
    count = 0
    for arg in sys.argv:
        if count > 2:
            train_spls.append(int(arg))
        count += 1
    nbc = NaiveBayesClassifier(filename=fname, num_splits=int(n_splits), train_percents=train_spls)
    testerror = nbc.model()
    print('Naive Bayes with marginal Gaussian distribution of {}\n'.format(nbc.filename))
    for p in nbc.train_percents:
        print('Mean test error for training percent of {} : {}\n'.format(p, testerror[p]*100))
