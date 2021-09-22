import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from examples.time_series.ts_forecasting_composing import plot_results
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum

DATA_TYPES = ['GS10', 'EXCHUS', 'EXCAUS',
              'Weekly U.S. Refiner and Blender Adjusted Net Production of Finished Motor Gasoline  (Thousand Barrels per Day)',
              'Weekly Minnesota Midgrade Conventional Retail Gasoline Prices  (Dollars per Gallon)',
              'Weekly U.S. Percent Utilization of Refinery Operable Capacity (Percent)',
              'Weekly U.S. Exports of Crude Oil and Petroleum Products  (Thousand Barrels per Day)',
              'Weekly U.S. Field Production of Crude Oil  (Thousand Barrels per Day)',
              'Weekly U.S. Ending Stocks of Crude Oil and Petroleum Products  (Thousand Barrels)',
              'Weekly U.S. Product Supplied of Finished Motor Gasoline  (Thousand Barrels per Day)']


def prepare_data(task, label="GS10", shift=0):
    long_df = pd.read_csv('../../data/ts_short.csv')
    long_df['datetime'] = pd.to_datetime(long_df['datetime'])
    df = long_df[long_df['series_id'] == label]

    data = np.array(df['value'])
    data = data[:data.shape[0] - shift]
    input_data = InputData(idx=np.arange(0, len(data)),
                           features=data,
                           target=data,
                           task=task,
                           data_type=DataTypesEnum.ts)

    return input_data


def create_plot(actual_time_series, predicted_values, len_train_data, y_name='Sea surface height, m'):
    fig = plt.figure(figsize=(20, 50))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, len(actual_time_series)),
            actual_time_series, label='Actual values', c='green')
    ax.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
            predicted_values, label='Predicted', c='blue')

    # Plot black line which divide our array into train and test
    ax.plot([len_train_data, len_train_data],
            [min(actual_time_series), max(actual_time_series)], c='black',
            linewidth=1)
    ax.set_xlabel(y_name, fontsize=15)
    ax.set_ylabel('Time index', fontsize=15)
    ax.legend(fontsize=15)
    return fig


def save_results(label, observed_data, predicted, target, pipeline, fig):
    new_dir = f"{label}_{len(target)}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.mkdir(new_dir)
    np.savetxt(os.path.join(new_dir, "observed_data.txt"), observed_data)
    np.savetxt(os.path.join(new_dir, "predicted_data.txt"), predicted)
    np.savetxt(os.path.join(new_dir, "real_target.txt"), target)

    pipeline.save(os.path.join(new_dir, "pipeline"))
    fig.savefig(os.path.join(new_dir, "plot.svg"))

    return new_dir


def evaluate_fedot(label, forecast_length, shift=0):
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    input_data = prepare_data(task, label=label, shift=shift)
    train_input, predict_input = train_test_data_setup(input_data)
    # Init model for the time series forecasting
    model = Fedot(problem='ts_forecasting', task_params=task.task_params, timeout=1)

    # Run AutoML model design in the same way
    pipeline = model.fit(features=train_input)

    forecast = model.predict(features=predict_input)

    # Plot results
    figure = create_plot(actual_time_series=input_data.features,
                         predicted_values=forecast,
                         len_train_data=len(input_data.features) - forecast_length,
                         y_name=label)

    save_results(label, predict_input.features, forecast, predict_input.target, pipeline, figure)


if __name__ == '__main__':
    # for label in DATA_TYPES:
    #     for _ in range(5):
    #         for shift in [0, 12, 24, 36]:
    #             evaluate_fedot(label, 10, shift)
    label = "GS10"
    for _ in range(5):
        for shift in [0, 12, 24, 36]:
            evaluate_fedot(label, 10, shift)
