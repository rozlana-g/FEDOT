import configparser
import json
import os
import sys

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.template import PipelineTemplate
from fedot.core.repository.tasks import Task

tmp_task = Task


def extract_data_from_config_file(file):
    config = configparser.ConfigParser()
    if not os.path.exists(file):
        raise ValueError()
    config.read(file, encoding='utf-8')
    pipeline_description = config['DEFAULT']['pipeline_description']
    input_data = config['DEFAULT']['train_data']
    task = eval(config['DEFAULT']['task'])
    output_path = config['DEFAULT']['output_path']

    test_data_path = config['OPTIONAL'].get('test_data')

    return pipeline_description, input_data, task, test_data_path, output_path


def run_fedot(config_file):
    pipeline_description, train_data_path, task, test_data_path, \
    output_path = extract_data_from_config_file(config_file)

    pipeline = pipeline_from_json(pipeline_description)

    train_data = InputData.from_csv(file_path=train_data_path,
                                    task=task)

    pipeline.fit_from_scratch(train_data)

    if test_data_path:
        test_data = InputData.from_csv(test_data_path)
        pipeline.predict(test_data)

    pipeline.save(path=output_path)


def pipeline_from_json(json_str: str):
    json_dict = json.loads(json_str)
    pipeline = Pipeline()
    pipeline.nodes = []
    pipeline.template = PipelineTemplate(pipeline, pipeline.log)

    pipeline.template._extract_operations(json_dict, None)
    pipeline.template.convert_to_pipeline(pipeline.template.link_to_empty_pipeline, None)
    pipeline.template.depth = pipeline.template.link_to_empty_pipeline.depth

    return pipeline


if __name__ == '__main__':
    config_file = sys.argv[1]
    run_fedot(config_file)
