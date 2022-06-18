import os
from absl import logging
import config as pipeline_config
from project import utils
from project import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2

def run():
  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
  tfx_image = os.environ.get('*********', None)
  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config=metadata_config,
    tfx_image=tfx_image
  )


  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
    pipeline.create_pipeline(
      pipeline_name=utils.PIPELINE_NAME,
      pipeline_root=pipeline_config.PIPELINE_ROOT_GCS,
      data_path=utils._data_root,
      preprocessing_fn=utils.PREPROCESSING_FN,
      run_fn=utils.RUN_FN,
      train_args=trainer_pb2.TrainArgs(),
      eval_args=trainer_pb2.EvalArgs(),
      serving_model_dir=pipeline_config.SERVING_MODEL_DIR_GCS,