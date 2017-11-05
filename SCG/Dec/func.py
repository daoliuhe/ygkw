"""
14/12/2016
    a. col_index_new: 62 candidate variables
    b. forward-backward filter is applied to vibration and inputs
    c. defining function can skip saving data file
    d. remove low r and var features

21/12/2016
***** Functions are defined here *****

"""

import pickle
from User.functions import slice_data, pcc, variance
import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


def prepare_data_(y, X, train_size):
    """ Prepare training and testing data, including external X """

    train_from = 20  # bigger than max lag, not smart way
    lags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # list not work
    n_y_input = len(lags)
    n_x_input = X.shape[1]
    n_input = n_y_input + n_x_input

    trainY = y[train_from:train_size]
    testY = y[train_size:]
    trainX = np.zeros((len(trainY), n_input))
    testX = np.zeros((len(testY), n_input))

    h1 = range(train_from, train_size)
    h2 = range(train_size, len(y))
    for i in range(0, n_input):
        if i < len(lags):
            trainX[:, i] = y[h1 - lags[i]]
            testX[:, i] = y[h2 - lags[i]]
        else:
            trainX[:, i] = X[:, i - n_y_input][h1 - lags[0]]
            testX[:, i] = X[:, i - n_y_input][h2 - lags[0]]

    return trainX, trainY, testX, testY


def iterative(model, X, H):
    """
        model: forecasting model
        X: input batch
        H: horizon
    """

    forecast = np.zeros(H)
    forecast[0] = model.predict(X[0, :].reshape(1, -1))  # 1-step ahead forecast

    for h in range(1, H):
        features = np.hstack((forecast[:h], X[h, :(10-h)], X[h, 10:]))  # 10 = no. of vibration lags
        forecast[h] = model.predict(features.reshape(1, -1))

    return forecast


def print_progress(step, max_step, stride=10):
    if step % stride == 0:
        print('Progress: %d/%d' % (step, max_step))


def mimo_data(series, window, H):
    """ data preparation for mimo model """

    n = len(series)
    n_samples = n - H - window + 1
    inputs = np.zeros((n_samples, window))
    targets = np.zeros((n_samples, H))

    for t in range(n_samples):
        inputs[t, :] = series[t:(t + window)]
        targets[t, :] = series[(t + window):(t + window + H)]

    return inputs, targets


def iterative_forecast(model, X, H, window):
    """
        model: forecasting model
        X: input batch
        H: horizon
        window: no. of features
    """

    forecast = np.zeros(H)
    forecast[0] = model.predict(X[0, :].reshape(1, -1))  # 1-step ahead forecast

    for h in range(1, H):
        augmented = np.hstack((forecast[:h], X[h, :]))
        features = augmented[:window]
        forecast[h] = model.predict(features.reshape(1, -1))

    return forecast


def direct_forecast(model, trainX, trainY, testX, H):
    """
        model: forecasting model
        trainX: newly updated X
        trainY: newly update Y
        testX: test X
        H: horizon
    """

    forecast = np.zeros(H)
    n = len(trainY)
    n_samples = n - H + 1

    for h in range(H):
        X_new = trainX[0:n_samples, :]
        Y_new = trainY[(0+h):(n_samples+h)]
        model.fit(X_new, Y_new)
        forecast[h] = model.predict(testX.reshape(1, -1))

    return forecast


def prepare_data(vib, train_size):
    """ Prepare training and testing data """

    train_from = 20  # bigger than max lag, not smart way
    lags = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # list not work
    n_features = len(lags)

    trainY = vib[train_from:train_size]
    testY = vib[train_size:]
    trainX = np.zeros((len(trainY), n_features))
    testX = np.zeros((len(testY), n_features))

    h1 = range(train_from, train_size)
    h2 = range(train_size, len(vib))
    for i in range(0, n_features):
        trainX[:, i] = vib[h1 - lags[i]]
        testX[:, i] = vib[h2 - lags[i]]

    return trainX, trainY, testX, testY


def remove_low_r_var(target, inputs):
    """ Remove features with low r and var """

    col_ind = pickle.load(open("../Nov/Regression/col_index_new.p", "rb"))

    # remove features when r < 0.1
    r_limit = 0.1
    r = pcc(inputs, target)
    r_ind = r >= r_limit
    inputs1 = inputs[:, r_ind]
    col_ind1 = np.array(col_ind)[r_ind]

    # remove features when var < 0.005, do scaling first
    sx = MinMaxScaler()
    inputs2 = sx.fit_transform(inputs1)
    var = variance(inputs2)
    var_limit = 0.005
    var_ind = var >= var_limit
    col_ind2 = col_ind1[var_ind]

    return list(col_ind2)


def filter_data(filter_size=5, start='14/Feb/16 00:10:00', end='25/Mar/16 00:10:00'):
    """ Apply forward-backward linear filter """

    data_sampled = pickle.load(open("../Data/sampled2016.p", "rb"))
    date_sampled = pickle.load(open("../Data/date2016.p", "rb"))
    data = slice_data(date_sampled, data_sampled, start, end)

    vib = data[:, 41]  # XI392-04A
    # filter_size = 9
    b = (1 / filter_size) * np.ones(filter_size)
    a = 1
    y = signal.filtfilt(b, a, vib)

    col_ind = pickle.load(open("../Nov/Regression/col_index_new.p", "rb"))
    inputs = data[:, col_ind]
    X = signal.filtfilt(b, a, inputs, axis=0)

    return y, X

