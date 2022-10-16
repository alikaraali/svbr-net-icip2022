import tensorflow as tf
import sys
import time
import numpy as np
import cv2

from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers as L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model



def read_tfrecord_new(serialized_example):
    """_summary_

    Args:
        serialized_example (_type_): _description_

    Returns:
        _type_: _description_
    """
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.string),
        'bmap1': tf.io.FixedLenFeature((), tf.string),
        'bmap2': tf.io.FixedLenFeature((), tf.string),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=tf.uint8)
    label = tf.io.parse_tensor(example['label'], out_type=tf.uint8)
    bmap1 = tf.io.parse_tensor(example['bmap1'], out_type=tf.uint8)
    bmap2 = tf.io.parse_tensor(example['bmap2'], out_type=tf.uint8)

    image_shape = [example['height'], example['width'], example['depth']]

    imaged = tf.reshape(image, image_shape)
    labeld = tf.reshape(label, image_shape)
    bmap1d = tf.reshape(bmap1, (image_shape[0], image_shape[1], 1))
    bmap2d = tf.reshape(bmap2, (image_shape[0], image_shape[1], 1))

    image = tf.math.divide(tf.cast(imaged, dtype=tf.float32), 255)
    bmap1 = tf.math.divide(tf.math.divide(tf.cast(bmap1d, dtype=tf.float32), 12), 6)
    bmap2 = tf.math.divide(tf.math.divide(tf.cast(bmap2d, dtype=tf.float32), 12), 6)
    label = tf.math.divide(tf.cast(labeld, dtype=tf.float32), 255)

    return image, label, bmap1, bmap2


