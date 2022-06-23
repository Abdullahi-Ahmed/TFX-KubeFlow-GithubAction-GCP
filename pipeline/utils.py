import os

PIPELINE_NAME = 'tfx_fashion_pipeline'

GCS_BUCKET_NAME = 'fashion-bucket254'

GCP_PROJECT_ID = 'dev-country-353212'
GCP_REGION = 'us-west1'

PREPROCESSING_FN = 'project.preprocess.preprocessing_fn'
RUN_FN = 'project.model.run_fn'
TUNER_FN = 'project.hptuner.tuner_fn'

_LABEL_KEY = 'label'
_IMAGE_KEY = 'image'


_data_root = 'my-bucket-34d9e85/data/fmnist'
