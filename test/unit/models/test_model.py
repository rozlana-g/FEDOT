import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score as roc_auc, mean_squared_error

from test.unit.tasks.test_regression import get_synthetic_regression_data
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.model import Model
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository


def get_roc_auc(valid_data: InputData, predicted_data: OutputData) -> float:
    n_classes = valid_data.num_classes
    if n_classes > 2:
        additional_params = {'multi_class': 'ovo', 'average': 'macro'}
    else:
        additional_params = {}

    try:
        roc_on_train = round(roc_auc(valid_data.target,
                                     predicted_data.predict,
                                     **additional_params), 3)
    except Exception as ex:
        print(ex)
        roc_on_train = 0.5

    return roc_on_train


@pytest.fixture()
def classification_dataset():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    threshold = 0.5
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = InputData(features=x, target=classes, idx=np.arange(0, len(x)),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)

    return data


def classification_dataset_with_redunant_features(
        n_samples=1000, n_features=100, n_informative=5) -> InputData:
    synthetic_data = make_classification(n_samples=n_samples,
                                         n_features=n_features,
                                         n_informative=n_informative)

    input_data = InputData(idx=np.arange(0, len(synthetic_data[1])),
                           features=synthetic_data[0],
                           target=synthetic_data[1],
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    return input_data


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_classification_models_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)
    roc_threshold = 0.95

    with OperationTypesRepository() as repo:
        model_names, _ = repo.suitable_operation(task_type=TaskTypesEnum.classification,
                                                 tags=['ml'])

    for model_name in model_names:
        model = Model(operation_type=model_name)
        _, train_predicted = model.fit(data=train_data)
        test_pred = model.predict(fitted_operation=_, data=test_data, is_fit_chain_stage=False)
        roc_on_test = get_roc_auc(valid_data=test_data,
                                  predicted_data=test_pred)
        assert roc_on_test >= roc_threshold


def test_regression_models_fit_correct():
    data = get_synthetic_regression_data(n_samples=1000, random_state=42)
    train_data, test_data = train_test_data_setup(data)

    with OperationTypesRepository() as repo:
        model_names, _ = repo.suitable_operation(task_type=TaskTypesEnum.regression,
                                                 tags=['ml'])

    for model_name in model_names:
        model = Model(operation_type=model_name)
        _, train_predicted = model.fit(data=train_data)
        test_pred = model.predict(fitted_operation=_, data=test_data, is_fit_chain_stage=False)
        rmse_value_test = mean_squared_error(y_true=test_data.target, y_pred=test_pred.predict)

        rmse_threshold = np.std(test_data.target) ** 2
        assert rmse_value_test < rmse_threshold


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_log_clustering_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Scaling chain. Fit predict it
    scaling_chain = Chain(PrimaryNode('normalization'))
    scaling_chain.fit(train_data)
    scaled_data = scaling_chain.predict(train_data)

    kmeans = Model(operation_type='kmeans')
    _, train_predicted = kmeans.fit(data=scaled_data)

    assert all(np.unique(train_predicted.predict) == [0, 1])


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_svc_fit_correct(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Scaling chain. Fit predict it
    scaling_chain = Chain(PrimaryNode('normalization'))
    scaling_chain.fit(train_data)
    scaled_data = scaling_chain.predict(train_data)

    svc = Model(operation_type='svc')
    _, train_predicted = svc.fit(data=scaled_data)

    roc_on_train = get_roc_auc(valid_data=train_data,
                               predicted_data=train_predicted)
    roc_threshold = 0.95
    assert roc_on_train >= roc_threshold


def test_pca_model_removes_redunant_features_correct():
    n_informative = 5
    data = classification_dataset_with_redunant_features(n_samples=1000, n_features=100,
                                                         n_informative=n_informative)
    train_data, test_data = train_test_data_setup(data=data)

    # Scaling chain. Fit predict it
    scaling_chain = Chain(PrimaryNode('normalization'))
    scaling_chain.fit(train_data)
    scaled_data = scaling_chain.predict(train_data)

    pca = DataOperation(operation_type='pca')
    _, train_predicted = pca.fit(data=scaled_data)
    transformed_features = train_predicted.predict

    assert transformed_features.shape[1] < data.features.shape[1]
