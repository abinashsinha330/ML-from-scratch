import pandas as pd
import numpy as np


def normalize(x):
    """In this method, we normalize the variable using the mean and standard deviation because we want the values of
    each feature to be on similar scales. It can happen one of the features has values in a higher range and the
    other has values in smaller range. In order, to bring them on the same scale we do normalization of the features"""
    feature_df = x.loc[:, 1:]
    return (feature_df - feature_df.mean())/feature_df.std()


def delete_useless_features(x):
    """Those features which have zero standard deviation do not contribute much to the training. Thus, delete them"""
    return x.drop(x.std()[x.std() == 0].index.values, axis=1)


def plot(axis, c, prop):
    axis.semilogx(c, prop)


def group_by_class(input_data):
    grouped_input_data = input_data.groupby(input_data.iloc[:, 0])
    return grouped_input_data


def random_class_specific_split(x):
    """Function that performs random class-specific 80-20 split i.e. 80% of data belonging to each class is taken
    together to form training data and 20% of data belonging to each class taken together to form test data"""
    grouped_data = group_by_class(x)
    classes = sorted(list(x.iloc[:, 0].unique()))
    train_data = pd.DataFrame()
    tst_data = pd.DataFrame()
    for c in classes:
        class_c_data = grouped_data.get_group(c)
        temp_data = class_c_data.sample(frac=0.20)
        tst_data = pd.concat((tst_data, temp_data), ignore_index=True)
        train_data = pd.concat((train_data, class_c_data.append(temp_data).drop_duplicates(keep=False)),
                               ignore_index=True)

    return train_data, tst_data


def calculate_error(predicts, actuals):
    num_errors = np.array([predicts[k] != actuals[k] for k in range(0, actuals.shape[0])])
    error_fraction = np.sum(num_errors) / actuals.shape[0]
    return error_fraction
