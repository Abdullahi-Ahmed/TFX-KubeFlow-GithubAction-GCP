import os
from typing import Optional, Text, List, Dict, Any

import tensorflow_model_analysis as tfma
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components import Tuner
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration import pipeline
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from ml_metadata.proto import metadata_store_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input

def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    preprocessing_fn: Text,
    tuner_fn:Text,
    run_fn: Text,
    serving_model_dir: Text,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
) -> pipeline.Pipeline:
    examples = external_input(data_path)
    input_split = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='fashion_mnist-train.tfrecord-00000-of-00001'),
        example_gen_pb2.Input.Split(name='eval', pattern='fashion_mnist-test.tfrecord-00000-of-00001')
    
    ])
    example_gen = ImportExampleGen(input=examples, input_config=input_split)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    infer_schema = SchemaGen(statistics=statistics_gen.outputs['statistics'], 
        infer_feature_shape=True)
    validate_stats = ExampleValidator(statistics=statistics_gen.outputs['statistics'],
        schema=infer_schema.outputs['schema'])
    transform = Transform(
        examples=example_gen.outputs['examples'], 
        schema=infer_schema.outputs['schema'],
        preprocessing_fn=preprocessing_fn)
                        
    tuner = Tuner(
        tuner_fn=tuner_fn,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema = infer_schema.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=500),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=100)
    )

    trainer = Trainer(
        run_fn=run_fn, 
        transformed_examples=transform.outputs['transformed_examples'],
        schema=infer_schema.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000)
                      )

    
    eval_config = tfma.EvalConfig(
        slicing_specs=[tfma.SlicingSpec()]
    )

    evaluator = Evaluator(examples=example_gen.outputs['examples'], 
        model=trainer.outputs['model'],
        eval_config=eval_config)

    validator = ModelValidator(examples=example_gen.outputs['examples'], 
        model=trainer.outputs['model']) 

    pusher = Pusher(model=trainer.outputs['model'], 
        model_blessing=validator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=serving_model_dir)))

    return pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=pipeline_root,
    components=[
      example_gen,
      statistics_gen,
      infer_schema,
      validate_stats,
      transform,
      tuner,
      trainer,
      evaluator,
      validator,
      pusher
    ],
    enable_cache=True,
    metadata_connection_config=metadata_connection_config,
    beam_pipeline_args=beam_pipeline_args,
   )
#        #Running Kubeflow Locally
# if __name__ == '__main__':
#     tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)

#     metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
#     metadata_config.mysql_db_service_host.value = 'mysql.kubeflow'
#     metadata_config.mysql_db_service_port.value = "3306"
#     metadata_config.mysql_db_name.value = "metadb"
#     metadata_config.mysql_db_user.value = "root"
#     metadata_config.mysql_db_password.value = ""
#     metadata_config.grpc_config.grpc_service_host.value = 'metadata-grpc-service'
#     metadata_config.grpc_config.grpc_service_port.value = '8080'

