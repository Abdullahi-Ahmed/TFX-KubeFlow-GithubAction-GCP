import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from . import preprocess
from . import model

LABEL_KEY = 'label_xf'
train_batch_size = 32
eval_batch_size = 32

def _gzip_reader_fn(filenames):
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _serving_input_receiver_fn(tf_transform_output):
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_feature_spec.pop(LABEL_KEY)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec,
                                                                               default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    transformed_features = tf_transform_output.transform_raw_features(serving_input_receiver.features)
    transformed_features.pop(preprocess._transformed_name(LABEL_KEY))

    return tf.estimator.export.ServingInputReceiver(transformed_features, serving_input_receiver.receiver_tensors)

def _eval_input_receiver_fn(tf_transform_output):
    raw_feature_spec = tf_transform_output.raw_feature_spec()

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(raw_feature_spec,
                                                                               default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    transformed_features = tf_transform_output.transform_raw_features(serving_input_receiver.features)
    transformed_labels = transformed_features.pop(preprocess._transformed_name(LABEL_KEY))

    return tfma.export.EvalInputReceiver(features=transformed_features, labels=transformed_labels,
                                         receiver_tensors=serving_input_receiver.receiver_tensors)

def _input_fn(filenames, tf_transform_output, batch_size):
    transformed_feature_spec = (tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(filenames, batch_size, transformed_feature_spec,
                                                                 reader=_gzip_reader_fn)

    return dataset.map(lambda features: (features, features.pop(preprocess._transformed_name(LABEL_KEY))))

def trainer_fn(trainer_fn_args, schema):  # pylint: disable=unused-argument
    tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

    train_input_fn = lambda: _input_fn(trainer_fn_args.train_files, tf_transform_output, batch_size=train_batch_size)

    eval_input_fn = lambda: _input_fn(trainer_fn_args.eval_files, tf_transform_output, batch_size=eval_batch_size)

    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=trainer_fn_args.train_steps)

    serving_receiver_fn = lambda: _serving_input_receiver_fn(tf_transform_output)

    exporter = tf.estimator.FinalExporter('cifar-10', serving_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=trainer_fn_args.eval_steps, exporters=[exporter],
                                      name='cifar-10')

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=999, keep_checkpoint_max=1)

    run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model.model_builder(), config=run_config)

    eval_receiver_fn = lambda: _eval_input_receiver_fn(tf_transform_output)

    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': eval_receiver_fn
    }