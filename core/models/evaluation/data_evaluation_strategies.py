import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

from core.models.data import InputData


def get_data(predict_data: InputData):
    return predict_data.features


def get_difference(predict_data: InputData):
    number_of_inputs = predict_data.features.shape[1]
    if number_of_inputs != 1:
        raise ValueError('Too many inputs for the differential model')
    return predict_data.features[:, 0] - predict_data.target


def get_sum(predict_data: InputData):
    if predict_data.features.shape[1] != 2:
        raise ValueError('Wrong number of inputs for the additive model')
    return np.sum(predict_data.features, axis=1)


def _estimate_period(variable):
    analyse_ratio = 10
    f, pxx_den = signal.welch(variable, fs=1, scaling='spectrum',
                              nfft=int(len(variable) / analyse_ratio),
                              nperseg=int(len(variable) / analyse_ratio))
    period = int(1 / f[np.argmax(pxx_den)])
    return period


def get_trend(predict_data: InputData):
    target = predict_data.target
    period = _estimate_period(target)
    decomposed_target = seasonal_decompose(target, period=period, extrapolate_trend='freq')
    return decomposed_target.trend


def get_residual(predict_data: InputData):
    target_trend = get_trend(predict_data)
    target_residual = predict_data.target - target_trend
    return target_residual
