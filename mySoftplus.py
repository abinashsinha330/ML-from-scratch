import numpy as np
import sys
import pandas as pd
import time
import math as m
import myPegasos
import matplotlib.pyplot as plt
import utility as util
import math


class SSVM(myPegasos.PSVM):
    """"""
    def __init__(self, x, y, param_lambda, k):
        super(SSVM, self).__init__(x, y, param_lambda, k)
        # self.x = x
        # self.y = y
        # self.param_lambda = param_lambda
        # self.k = k
        self.primal_values = []
        self.a = 1

    def primal_obj_fn(self, x, y, param_w, param_lambda, num_examples):
        softplus = self.a * np.log(1 + np.exp((1 - (np.dot(x, param_w) * y)) / self.a))
        softplus_sum = np.sum(softplus)
        reg = 0.5 * param_lambda * param_w.T.dot(param_w)[0, 0]
        primal_value = reg + ((1 / num_examples) * softplus_sum)
        return primal_value

    def model(self):
        x = self.x
        y = self.y
        param_lambda = self.param_lambda
        k = self.k
        num_features = x.shape[1]
        num_examples = x.shape[0]
        init_w = np.expand_dims(np.zeros(num_features), axis=1)
        init_w.fill(np.sqrt(1 / param_lambda / num_features))
        w_t = init_w
        num_iter = 100 * num_examples + 1
        count = 0
        for t in range(1, num_iter+1):
            batch_x_t, batch_y_t = SSVM.k_sample(x, y, k)
            eta_k = 1.0 / (param_lambda * float(t))
            summation = np.zeros([num_features, 1])
            for index in range(batch_x_t.shape[0]):
                batch_x_t_trans = np.expand_dims(batch_x_t[index, :].T, axis=1)
                numerator = batch_y_t[index, 0] * batch_x_t_trans
                denominator = 1 + np.exp((batch_y_t[index, 0] * np.dot(w_t.T, batch_x_t_trans)[0, 0] - 1) / self.a)
                summation += (numerator / denominator)

            param_w_t_plus_1by2 = ((1 - eta_k * param_lambda) * w_t) + \
                                  ((eta_k / k) * summation)

            minimum = min(1.0, ((1/m.sqrt(param_lambda))/np.linalg.norm(param_w_t_plus_1by2)))
            w_t_plus_1 = minimum * param_w_t_plus_1by2
            primal_value = self.primal_obj_fn(x, y, w_t_plus_1, param_lambda, num_examples)
            self.primal_values.append(primal_value)
            count += 1
            if (np.sum((w_t_plus_1 - w_t) ** 2)/w_t.shape[0]) < 0.000001:
                break
            else:
                w_t = w_t_plus_1
        predict = SSVM.predict(w_t, 0, x, y)
        return w_t, count, predict


if __name__ == "__main__":
    filename = sys.argv[1]
    ncols = 2
    params_k = []
    num_runs = int(sys.argv[2])
    count = 0
    for arg in sys.argv:
        if count > 2:
            params_k.append(int(arg))
        count += 1
    data = pd.read_csv(filename, header=None, index_col=False)
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == 3:
            data.iloc[i, 0] = -1
    data = util.delete_useless_features(data)
    data.loc[:, 1:] = util.normalize(data)
    x_array = np.array(data.loc[:, 1:])
    y_array = np.expand_dims(np.array(data.loc[:, 0]), axis=1)

    runtimes = {}
    iterations = {}
    errors = {}

    print('\nStatistics of softplus algorithm for SVM')
    for i in range(num_runs):
        nrows = int(math.ceil(len(params_k) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
        row_i = 0
        col_i = 0
        k = 0
        for param_k in params_k:
            print('k value: ', param_k)
            if param_k not in runtimes:
                runtimes[param_k] = []
            if param_k not in iterations:
                iterations[param_k] = []
            if param_k not in errors:
                errors[param_k] = []
            start = time.time()
            ssvm = SSVM(x_array, y_array, 1e-4, param_k)
            w, iterations_run, predicts = ssvm.model()
            finish = time.time()
            runtimes[param_k].append(finish - start)
            iterations[param_k].append(iterations_run)
            errors[param_k].append(util.calculate_error(predicts, y_array))
            if col_i == ncols:
                col_i = 0
                row_i += 1
            axes[row_i, col_i].set_title('Softplus Loss vs Number of iterations \nfor mini-batch size,'
                                         ' k = {}'.format(param_k))
            axes[row_i, col_i].plot(ssvm.primal_values)
            col_i += 1
            k += 1

        plt.tight_layout()
        plt.savefig('softplus_run_{}_plots.pdf'.format(i), transparent=True, dpi=600)
        plt.title('Softplus loss vs Number of iterations for run {}'.format(i+1))

    # printing out the mean and standard deviation of runtimes, errors and iterations run to converge
    for k in params_k:
        print('\nMean runtime for mini-batch size, k = {} in seconds: {}'.format(k, np.mean(runtimes[k])))
        print('\nStandard deviation of runtime for mini-batch size, k = {} in seconds: {}'
              .format(k, np.std(runtimes[k])))
        print('\nMean number of iterations run for mini-batch size, k = {} to converge: {}'
              .format(k, int(np.mean(iterations[k]))))
        print('\nMean of percentage errors for mini-batch size, k = {}: {}'
              .format(k, np.mean(errors[k]) * 100))
        print('\nStandard deviation of percentage errors for mini-batch size, k = {}: {}'
              .format(k, np.std(errors[k]) * 100))
