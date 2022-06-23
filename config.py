import os
from pipeline import utils



OUTPUT_DIR = os.path.join(".", "tfx")
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             utils.PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_DIR, 'tfx_metadata', utils.PIPELINE_NAME,
                             'metadata.db')

SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")


OUTPUT_DIR_GCS = os.path.join('gs://', utils.GCS_BUCKET_NAME)
PIPELINE_ROOT_GCS = os.path.join(OUTPUT_DIR_GCS, 'tfx_pipeline_output',
                                 utils.PIPELINE_NAME)
SERVING_MODEL_DIR_GCS = os.path.join(PIPELINE_ROOT_GCS, 'serving_model')

