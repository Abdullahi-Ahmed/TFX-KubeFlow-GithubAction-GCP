import tensorflow as tf
import tensorflow_transform as tft

from . import utils

def _transformed_name(key):
    return key + '_xf'

def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _image_parser(image_str):
    '''converts the images to a float tensor'''
    image = tf.image.decode_image(image_str, channels=1)
    image = tf.reshape(image, (28, 28, 1))
    image = tf.cast(image, tf.float32)
    return image

def _label_parser(label_id):
    '''converts the labels to a float tensor'''
    label = tf.cast(label_id, tf.float32)
    return label


def preprocessing_fn(inputs):
    # Convert the raw image and labels to a float array
    with tf.device("/cpu:0"):
        outputs = {
            _transformed_name(utils._IMAGE_KEY):
                tf.map_fn(
                    _image_parser,
                    tf.squeeze(inputs[utils._IMAGE_KEY], axis=1),
                    dtype=tf.float32),
            _transformed_name(utils._LABEL_KEY):
                tf.map_fn(
                    _label_parser,
                    inputs[utils._LABEL_KEY],
                    dtype=tf.float32)
        }
    
    # scale the pixels from 0 to 1
    outputs[_transformed_name(utils._IMAGE_KEY)] = tft.scale_to_0_1(outputs[_transformed_name(utils._IMAGE_KEY)])
    
    return outputs