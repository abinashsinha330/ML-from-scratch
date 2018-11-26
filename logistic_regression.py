import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys

class LogisticRegression:
    """This class is for performing logistic regression using Iteratively Reweighted Least Squares Algorithm"""

    def __init__(self, filename, num_splits=10, num_iter=10, train_percents=[10, 25, 50, 75, 100]):
        if not filename:
            raise ValueError('Please provide a file to retrieve data from')
        self.filename = filename
        self.data = self.preprocess(pd.read_csv(filename, header=None, index_col=False), filename)
        self.num_splits = num_splits
        self.num_iter = num_iter
        self.training_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.classes = sorted(list(self.data.iloc[:, -1].unique()))
        self.train_percents = train_percents

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
    def calculate_error(input_data, labels):
        errors = np.sum(labels != input_data.iloc[:, -1])
        return errors

    @staticmethod
    def drop_lastcol(input_data):
        return input_data.drop(input_data.columns[[-1]], axis=1)

    @staticmethod
    def groupby_class(input_data):
        grouped_input_data = input_data.groupby(input_data.iloc[:, -1])
        return grouped_input_data

    @staticmethod
    def get_y_vector(trainin_target_values, classes_without_last):
        y = []
        for c in classes_without_last:
            for a in trainin_target_values:
                if a == c:
                    y.append(1)
                else:
                    y.append(0)
        return y

    def random_class_specific_split(self):
        """Function that performs random class-specific 80-20 split i.e. 80% of data belonging to each class is taken
        together to form training data and 20% of data belonging to each class taken together to form test data"""
        grouped_data = LogisticRegression.groupby_class(self.data)
        training_data = pd.DataFrame()
        test_data = pd.DataFrame()
        for c in self.classes:
            class_c_data = grouped_data.get_group(c)
            temp_data = class_c_data.sample(frac=0.20)
            test_data = pd.concat((test_data, temp_data), ignore_index=True)
            training_data = pd.concat((training_data, class_c_data.append(temp_data).drop_duplicates(keep=False)),
                                      ignore_index=True)

        return training_data, test_data

    def get_percent_train_data(self, p, whole_train_data):
        """Function to give the p % of whole data which would be used for training our logistic regression"""
        grouped_data = LogisticRegression.groupby_class(whole_train_data)
        percent_training_data = pd.DataFrame()
        for c in self.classes:
            class_c_data = grouped_data.get_group(c)
            temp_data = class_c_data.sample(frac=p)
            percent_training_data = pd.concat((percent_training_data, temp_data), ignore_index=True)

        return percent_training_data

    def form_r_aggr(self, classes_without_last, len_training_data, r_aggr, x, w_vectors_map):
        for a in classes_without_last:
            r_aggr_row = np.zeros((len_training_data, len_training_data))
            for b in classes_without_last:
                if b == 0:
                    if a == b:
                        p = self.probability_y_given_x(x, a, w_vectors_map, len_training_data)
                        r_aggr_row += np.diag((p * (1 - p)).ravel())  # element wise product
                    else:
                        p = self.probability_y_given_x(x, a, w_vectors_map, len_training_data)
                        q = self.probability_y_given_x(x, b, w_vectors_map, len_training_data)
                        r_aggr_row += np.diag((p * q).ravel())
                else:
                    if a == b:
                        p = self.probability_y_given_x(x, a, w_vectors_map, len_training_data)
                        r_aggr_row = np.block([r_aggr_row, np.diag((p * (1 - p)).ravel())])
                    else:
                        p = self.probability_y_given_x(x, a, w_vectors_map, len_training_data)
                        q = self.probability_y_given_x(x, b, w_vectors_map, len_training_data)
                        r_aggr_row = np.block([r_aggr_row, np.diag((p * q).ravel())])

            if a == 0:
                r_aggr += r_aggr_row
            else:
                r_aggr = np.block([[r_aggr], [r_aggr_row]])
        return r_aggr

    def probability_y_given_x(self, x, c, w_vectors_map, length):
        """Function to get the probability that an N x D dimensional input, x belongs to class, c given the input,
        x using logistic function (where N is number of inputs)"""
        denominator = np.ones((1, length))
        classes_without_last = self.classes.copy()
        del classes_without_last[-1]
        for i in classes_without_last:
            denominator += np.exp(w_vectors_map[i].T.dot(x.T))
        numerator = np.exp(w_vectors_map[c].T.dot(x.T))
        return (numerator / denominator).T

    def set_w_vectors(self, w_aggr, dim_training_data):
        """Function to get the w vectors"""
        w_vectors_map = {}
        classes_without_last = self.classes.copy()
        del classes_without_last[-1]
        first_index = 0
        for c in classes_without_last:
            last_index = first_index + dim_training_data
            w_vectors_map[c] = w_aggr[first_index : last_index]
            first_index = last_index

        return w_vectors_map

    def form_p_aggr(self, classes_without_last, p_aggr, w_vectors_map, x, length):
        for c in classes_without_last:
            if c == 0:
                p_aggr += self.probability_y_given_x(x, c, w_vectors_map, length)
            else:
                res = self.probability_y_given_x(x, c, w_vectors_map, length)
                p_aggr = np.block([[p_aggr], [res]])
        return p_aggr

    def model(self):
        sum_errors = {}
        for percent in self.train_percents:
            sum_errors[percent] = 0
        classes_without_last = self.classes.copy()
        del classes_without_last[-1]
        num_classes = len(self.classes)
        if num_classes > 2:
            for i in range(self.num_splits):
                whole_training_data, test_data = self.random_class_specific_split()
                for percent in self.train_percents:
                    training_data = self.get_percent_train_data(percent/100.0, whole_training_data)
                    len_test_data = test_data.shape[0]
                    len_training_data = training_data.shape[0]
                    dim_training_data = training_data.shape[1] - 1

                    # input array of size N*(K-1) x D*(K-1) where N is number of training example and K is number of classes
                    identity_matrix = np.zeros((num_classes - 1, num_classes - 1))
                    x = LogisticRegression.drop_lastcol(training_data).values
                    x_aggr = np.kron(identity_matrix, x)
                    trainin_target_values = x[:, -1]
                    y_aggr = LogisticRegression.get_y_vector(trainin_target_values, classes_without_last)
                    w_aggr = np.zeros(((num_classes - 1) * dim_training_data, 1))
                    x_aggr_trans = x_aggr.T
                    for _ in range(self.num_iter):
                        w_vectors_map = self.set_w_vectors(w_aggr, dim_training_data)
                        temp1 = np.zeros((len_training_data, 1))
                        p_aggr = self.form_p_aggr(classes_without_last, temp1, w_vectors_map, x, len_training_data)
                        temp2 = np.zeros((len_training_data, len_training_data * (num_classes - 1)))
                        r_aggr = self.form_r_aggr(classes_without_last, len_training_data, temp2, x, w_vectors_map)
                        z_aggr = x_aggr.dot(w_aggr) - (np.linalg.pinv(r_aggr)).dot(p_aggr - y_aggr)
                        w_aggr = (
                            ((np.linalg.pinv((x_aggr_trans.dot(r_aggr)).dot(x_aggr))).dot(x_aggr_trans)).dot(r_aggr)) \
                            .dot(z_aggr)

                    w_vectors_map = self.set_w_vectors(w_aggr, dim_training_data)
                    labels_list = []
                    for j in range(len_test_data):
                        x_vector = LogisticRegression.drop_lastcol(test_data).iloc[[j]].T
                        max_p_c_given_x = 0
                        sum_p = 0
                        for c in classes_without_last:
                            p_c_given_x = self.probability_y_given_x(x_vector, c, w_vectors_map, len_training_data)
                            sum_p += p_c_given_x
                            if p_c_given_x > max_p_c_given_x:
                                max_p_c_given_x = p_c_given_x
                                label_x = c
                        if (1 - sum_p) > max_p_c_given_x:
                            label_x = self.classes[-1]
                        labels_list.append(label_x)

                    labels = np.array(labels_list)
                    test_error = LogisticRegression.calculate_error(test_data, labels) / float(len_test_data)
                    sum_errors[percent] += test_error
        else:
            for i in range(self.num_splits):
                whole_training_data, test_data = self.random_class_specific_split()
                for percent in self.train_percents:
                    training_data = self.get_percent_train_data(percent/100.0, whole_training_data)
                    len_test_data = test_data.shape[0]
                    dim_training_data = training_data.shape[1] - 1

                    x = LogisticRegression.drop_lastcol(training_data).values  # input 2D array of size N x D
                    y = np.expand_dims(training_data.values[:, -1], axis=1)
                    w = np.zeros((dim_training_data, 1))
                    x_trans = x.T
                    for _ in range(self.num_iter):
                        p = 1 / (1 + np.exp(-w.T.dot(x_trans))).T
                        r = np.diag((p * (1 - p)).ravel())  # element wise product
                        z = x.dot(w) - (np.linalg.pinv(r)).dot(p - y)
                        w = (((np.linalg.pinv((x_trans.dot(r)).dot(x))).dot(x_trans)).dot(r)).dot(z)

                    labels_list = []
                    for j in range(len_test_data):
                        x_vector = LogisticRegression.drop_lastcol(test_data).iloc[[j]].T
                        p_1_given_x = 1 / (1 + np.exp(-w.T.dot(x_vector)))
                        if p_1_given_x >= 0.5:
                            label_x = 1
                        else:
                            label_x = 0
                        labels_list.append(label_x)
                    labels = np.array(labels_list)
                    test_error = LogisticRegression.calculate_error(test_data, labels) / float(len_test_data)
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
        plt.title('Learning Curve (logistic regression for {} data)'.format(self.filename))
        # function to show the plot
        # plt.show()
        plt.savefig('Learning_Curve_{}_Logistic.png'.format(self.filename))


if __name__ == '__main__':
    fname = sys.argv[1]
    n_splits = sys.argv[2]
    n_iter = sys.argv[3]
    print(sys.argv)
    train_spls = []
    count = 0
    for arg in sys.argv:
        if count > 3:
            train_spls.append(int(arg))
        count += 1
    lr = LogisticRegression(filename=fname, num_splits=int(n_splits), num_iter=int(n_iter),
                            train_percents=train_spls)
    testerror = lr.model()
    print('Logistic regression of {}\n'.format(lr.filename))
    for p in lr.train_percents:
        print('Mean test error for training percent of {} : {}\n'.format(p, testerror[p]*100))
