import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import random
import sys


class LinearDiscriminantAnalysis:
    """This class is for conducting linear discriminant analysis for univariate and multi-variate scenarios. It also has
    utility functions to plot the data on 1-D or 2-D and also used for getting histogram plotting for univariate case.
    Also, the projected points on n-dimensional space is modelled using Gaussian distribution either univariate or multi
    variate as per the scenario and discriminant is found out.
    """

    def __init__(self, filename, output_dims, num_crossfolds):
        if not filename:
            raise ValueError('Please provide a file to retrieve data from')
        self.filename = filename
        self.data = self.preprocess(pd.read_csv(filename, header=None, index_col=False), filename)
        self.output_dims = output_dims
        self.num_crossfolds = num_crossfolds
        self.grouped_data = []
        self.classes = []
        self.cross_fold_serial = ''
        self.num_classes = len(sorted(list(self.data.iloc[:, -1].unique())))
        if self.output_dims == 0:
            raise ValueError('Please provide a non-zero dimension of subspace to project on')
        if  self.output_dims > self.num_classes-1:
            raise ValueError('LDA restricts projection of dataset on subspace having more than (K-1) dimension')
        self.grouped_data_map = {}
        self.p = {}
        self.num_features = self.data.shape[1] - 1  # subtracting 1 since there is additional column of target values
        self.w = np.zeros((output_dims, self.num_features))

    def model(self):
        if self.num_classes > 2:
            train_error, stdev_trainerror, test_error, stdev_testerror = self.lda_multiclass()
        else:
            train_error, stdev_trainerror, test_error, stdev_testerror = self.lda_2class()
            self.lda_plot_hist()
        return train_error, stdev_trainerror, test_error, stdev_testerror

    def lda_2class(self):
        """Function to use Fisher's Linear Discriminant analysis for 2 classes to find the direction of projection"""
        # mean vector of each class
        means = {}
        self.classes = sorted(list(self.data.iloc[:, -1].unique()))
        self.grouped_data = self.data.groupby(self.data.iloc[:, -1])
        for c in self.classes:
            self.grouped_data_map[c] = self.grouped_data.get_group(c)
        for c in self.classes:
            class_c_data = self.grouped_data.get_group(c)
            means[c] = np.expand_dims(np.array(class_c_data.drop(class_c_data.columns[[-1, ]], axis=1).mean(axis=0)),
                                      axis=1)
        s_w = np.zeros((self.num_features, self.num_features))

        for c in self.classes:
            class_c_data = self.drop_lastcol(self.grouped_data.get_group(c))
            class_c_variance = np.zeros((self.num_features, self.num_features))
            for j in range(class_c_data.shape[0]):
                row = class_c_data.iloc[[j]].transpose()
                class_c_variance += (row - means[c]).dot((row - means[c]).T)
            s_w += class_c_variance
        self.w = np.matmul(np.linalg.pinv(s_w), means[1]-means[0]).T
        if self.output_dims == 1:
            labels = self.univariate_gaussian_classifier(self.data)
        training_error = self.calculate_error(self.data, labels) / float(self.data.shape[0])

        return training_error, 0, 0, 0

    def kfold_split(self):
        """Function to split the dataframe into k folds randomly. It derives sample of given fold size randomly and then
        derives sample from remaining dataframe randomly and so on to finally get k folds
        """
        data_folds = dict()
        data_copy = self.data.copy()
        fold_size = int(self.data.shape[0] / self.num_crossfolds)
        for i in range(self.num_crossfolds):
            if i == (self.num_crossfolds - 1):
                data_folds[i] = data_copy
                break
            data_sample = data_copy.sample(fold_size)
            data_folds[i] = data_sample
            data_copy = data_copy.append(data_sample).drop_duplicates(keep=False)

        return data_folds

    def lda_multiclass(self):
        """Function to peform linear discriminant analysis for more than 2 classes"""
        sum_train_error = 0
        sum_test_error = 0
        data_folds = self.kfold_split()
        training_error_seq = []
        test_error_seq = []
        for i in range(self.num_crossfolds):
            self.cross_fold_serial = str(i+1)
            training_data = pd.DataFrame()  # empty dataframe
            test_data = pd.DataFrame()  # empty dataframe

            for key in data_folds:
                if key != i:
                    training_data = pd.concat((training_data, data_folds[key]), ignore_index=True)
            test_data = pd.concat((test_data, data_folds[i]), ignore_index=True)
            # mean vector of each class
            means = {}
            overall_mean = np.expand_dims(np.array(LinearDiscriminantAnalysis.drop_lastcol(training_data).mean(axis=0)),
                                          axis=1)
            self.classes = sorted(list(training_data.iloc[:, -1].unique()))
            self.grouped_data = training_data.groupby(training_data.iloc[:, -1])
            for c in self.classes:
                self.grouped_data_map[c] = self.grouped_data.get_group(c)
            s_b = np.zeros((self.num_features, self.num_features))
            for c in self.classes:
                class_c = self.grouped_data.get_group(c)
                means[c] = np.expand_dims(np.array(LinearDiscriminantAnalysis.drop_lastcol(class_c).mean(axis=0)),
                                          axis=1)
                count_c = class_c.shape[0]
                temp = np.multiply(count_c, np.matmul((means[c] - overall_mean), (means[c] - overall_mean).T))
                s_b += temp

            # s_w, within class covariance matrix calculation = variance of class 1 + variance of class 2
            s_w = np.zeros((self.num_features, self.num_features))
            for c in self.classes:
                class_c_data = self.drop_lastcol(self.grouped_data_map[c])
                class_c_variance = np.zeros((self.num_features, self.num_features))
                for j in range(class_c_data.shape[0]):
                    row = class_c_data.iloc[[j]].transpose()
                    class_c_variance += (row - means[c]).dot((row - means[c]).T)
                s_w += class_c_variance

                # pseudo inverse is used in case the inverse is not possible
                eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.pinv(s_w), s_b))
                eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]
                # sort the eigvals in decreasing order
                eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)
                # take the first eigvector which is the largest vector for projecting to 1-D
                self.w = np.array([eiglist[i][1] for i in range(self.output_dims)])

            if self.output_dims == 1:
                training_labels = self.univariate_gaussian_classifier(training_data)
                test_labels = self.univariate_gaussian_classifier(test_data)
            else:
                training_labels = self.multivariate_gaussian_classifier(training_data)
                test_labels = self.multivariate_gaussian_classifier(test_data)

            fold_train_error = (self.calculate_error(training_data, training_labels) / float(training_data.shape[0]))
            sum_train_error += fold_train_error
            training_error_seq.append(fold_train_error)
            fold_test_error = (self.calculate_error(test_data, test_labels) / float(test_data.shape[0]))
            sum_test_error += fold_test_error
            test_error_seq.append(fold_test_error)
            print('Training error for fold {}\n'.format(i+1), fold_train_error*100)
            print('Test error for fold {}\n'.format(i + 1), fold_test_error*100)
            self.lda_ndims_projection(training_data, self.output_dims)

        training_error = sum_train_error / self.num_crossfolds
        test_error = sum_test_error / self.num_crossfolds
        stddev_trainerror = ((np.sum((np.asarray(training_error_seq) - training_error)**2))/self.num_crossfolds)**0.5
        stdev_testerror = ((np.sum((np.asarray(test_error_seq) - test_error) ** 2)) / self.num_crossfolds)**0.5
        return training_error, stddev_trainerror, test_error, stdev_testerror

    @staticmethod
    def drop_lastcol(inp_data):
        return inp_data.drop(inp_data.columns[[-1]], axis=1)

    @staticmethod
    def univariate_gaussian_pdf(x, mean, covariance):
        cons = 1. / ((2 * np.pi) ** (len(x) / 2.) * covariance ** (-0.5))
        return cons * np.exp(-np.dot(np.dot((x - mean), covariance ** (-1)), (x - mean).T) / 2.)

    @staticmethod
    def multivariate_gaussian_pdf(x, mean, covariance):
        cons = 1. / ((2 * np.pi) ** (len(x) / 2.) * (np.linalg.det(covariance) ** (-0.5)))
        return cons * np.exp(-np.dot(np.dot((x - mean), np.linalg.pinv(covariance)), (x - mean).T) / 2.)

    def univariate_gaussian_classifier(self, train_data):
        """Function to perform gaussian modelling to find the discriminant"""

        classes = self.classes
        gaussian_means = {}
        gaussian_covariance = {}
        for c in classes:
            x_c = LinearDiscriminantAnalysis.drop_lastcol(self.grouped_data.get_group(c))
            y_c = np.dot(self.w, x_c.T).T
            # probability of each class occurence in the whole data i.e. p(c)
            self.p[c] = x_c.shape[0] / float(self.data.shape[0])
            gaussian_means[c] = np.mean(y_c, axis=0)
            gaussian_covariance[c] = np.cov(y_c, rowvar=False)

        inputs = LinearDiscriminantAnalysis.drop_lastcol(train_data)
        # project the inputs
        proj = np.dot(self.w, inputs.T).T
        # calculate the likelihoods for each class based on the gaussian models
        labels_list = []
        for x in proj:
            label_x = 0
            max_p_x_given_c = 0
            for c in classes:
                # we won't take into consideration prior probability of input, x since it will be common in all density
                # function
                p_x_given_c = self.p[c] * self.univariate_gaussian_pdf(x, gaussian_means[c],
                                                                       gaussian_covariance[c])
                if p_x_given_c > max_p_x_given_c:
                    max_p_x_given_c = p_x_given_c
                    label_x = c
            labels_list.append(label_x)
        labels = np.array(labels_list)

        return labels

    def multivariate_gaussian_classifier(self, train_data):
        """Function to perform gaussian modelling to find the discriminant"""

        classes = sorted(list(train_data.iloc[:, -1].unique()))
        group_data = train_data.groupby(train_data.iloc[:, -1])
        gaussian_means = {}
        gaussian_covariance = {}
        for c in classes:
            x_c = self.drop_lastcol(group_data.get_group(c))
            y_c = np.dot(self.w, x_c.T).T
            # probability of each class occurence in the whole data i.e. p(c)
            self.p[c] = x_c.shape[0] / float(train_data.shape[0])
            gaussian_means[c] = np.mean(y_c, axis=0)
            gaussian_covariance[c] = np.cov(y_c, rowvar=False)

        inputs = self.drop_lastcol(train_data)
        # project the inputs
        y = np.dot(self.w, inputs.T).T
        # calculate the likelihoods for each class based on the gaussian models
        labels_list = []
        for i in y:
            label_x = 0
            max_p_x_given_c = 0
            for c in classes:
                # we won't take into consideration prior probability of input, x since it will be common in all density
                # function [i[ind] for ind in range(len(i))]
                # print(type(i))
                p_x_given_c = self.p[c] * self.multivariate_gaussian_pdf(i,
                                                                         gaussian_means[c],
                                                                         gaussian_covariance[c])
                if p_x_given_c > max_p_x_given_c:
                    max_p_x_given_c = p_x_given_c
                    label_x = c
            labels_list.append(label_x)
        labels = np.array(labels_list)

        return labels

    @staticmethod
    def calculate_error(input_data, labels):
        errors = np.sum(labels != input_data.iloc[:, -1])
        return errors

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

    def lda_ndims_projection(self, data_given, output_dims):
        """Function to project data on n dimensions where output_dims gives the value of n (either of 1 or 2 only)"""
        classes = self.classes
        colors = cm.rainbow(np.linspace(0, 1, len(classes)))
        labels = {classes[c]: colors[c] for c in range(len(classes))}
        fig = plt.figure()
        for i, row in data_given.iterrows():
            proj = np.dot(self.w, row[:-1])
            if output_dims == 1:
                x = proj
                # adding a slight normally distributed noise for clear vision of overlapping of classes on 1-D line
                y = np.random.normal(0, 0.001, 1)
            else:
                x = proj[0]
                y = proj[1]
            plt.scatter(x, y, color=labels[row[data_given.shape[1] - 1]])
        title = ''
        if self.output_dims == 1:
            title = ' with slight normally distributed noise\n in y-axis values'
        plt.title('{}-D projection{} of {}'.format(self.output_dims, title, self.filename))
        self.cross_fold_serial
        fig.savefig('{}_projection{}.png'.format(self.filename, self.cross_fold_serial))

    def lda_plot_hist(self):
        """Function to plot histogram of projected points on 1-D after performing LDA"""
        num_bins = 20
        fig = plt.figure()
        # the histogram of the data
        for c in self.classes:
            x_c = self.drop_lastcol(self.grouped_data.get_group(c))
            y_c = np.dot(self.w, x_c.T).T
            plt.hist(y_c.T.tolist(), num_bins, alpha=0.5, label='Class {}'.format(str(int(c))))
            plt.legend(loc='upper right')
        plt.title('Histogram of Boston50 projected points')
        fig.savefig('Histogram_Boston50.png')


if __name__ == '__main__':
    fname = sys.argv[1]
    out_dim = sys.argv[2]
    n_folds = sys.argv[3]
    lda = LinearDiscriminantAnalysis(filename=fname, output_dims=int(out_dim),
                                     num_crossfolds=int(n_folds))
    mean_trainerror, stdev_train_error, mean_testerror, stdev_test_error = lda.model()

    if lda.num_classes > 2:
        print('{} % mean training error in digits recognition\n'.format(mean_trainerror * 100))
        print('{} % mean test error in digits recognition\n'.format(mean_testerror * 100))
        print('{} % standard deviation of training errors in digits recognition\n'.format(stdev_train_error * 100))
        print('{} % standard deviation of test errors in digits recognition\n'.format(stdev_test_error * 100))
    else:
        print('{} % training error using 100% of Boston50 data\n'.format(mean_trainerror * 100))
