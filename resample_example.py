import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from examples.classification_with_tuning_example import get_classification_dataset
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import SecondaryNode, PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task


def get_pipeline_with_balancing(custom_params=None):
    node_resample = PrimaryNode(operation_type='resample')

    if custom_params is not None:
        node_resample.custom_params = custom_params

    graph = SecondaryNode(operation_type='logit', nodes_from=[node_resample])

    return Pipeline(graph)


def get_pipeline_without_balancing():
    node = PrimaryNode(operation_type='logit')

    return Pipeline(node)


if __name__ == '__main__':
    samples = 1000
    features = 10
    classes = 2
    weights = [0.45, 0.55]
    features_options = {'informative': 1, 'redundant': 1, 'repeated': 1, 'clusters_per_class': 1}

    x_train, y_train, x_test, y_test = get_classification_dataset(features_options,
                                                                  samples,
                                                                  features,
                                                                  classes,
                                                                  weights)

    unique_class, counts_class = np.unique(y_train, return_counts=True)
    print(f'Two classes: {unique_class}')
    print(f'{unique_class[0]}: {counts_class[0]}')
    print(f'{unique_class[1]}: {counts_class[1]}')

    task = Task(TaskTypesEnum.classification)

    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    print(f'Begin fit Pipeline without balancing')
    # Pipeline without balancing
    pipeline = get_pipeline_without_balancing()

    # pipeline.fit(train_input)
    pipeline.fit_from_scratch(train_input)

    # Predict
    predict_labels = pipeline.predict(predict_input)
    preds = predict_labels.predict
    print('---')
    print(f"ROC-AUC of pipeline without balancing {roc_auc(y_test, preds):.4f}\n")

    # Pipeline with balancing
    pipeline = get_pipeline_with_balancing()

    print(f'Begin fit Pipeline with balancing')
    # pipeline.fit(train_input)
    pipeline.fit_from_scratch(train_input)

    # Predict
    predict_labels = pipeline.predict(predict_input)
    preds = predict_labels.predict
    print('---')
    print(f"ROC-AUC of pipeline with balancing {roc_auc(y_test, preds):.4f}\n")
