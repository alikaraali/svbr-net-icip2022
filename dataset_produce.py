import cv2
import numpy as np
# import matplotlib.pyplot as plt
import time
import tensorflow as tf

from utils_dataset_produce import propagate_laplacian, propagate_domaintransform



def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.string),
        'bmap': tf.io.FixedLenFeature((), tf.string),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64)
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=tf.uint8)
    label = tf.io.parse_tensor(example['label'], out_type=tf.uint8)
    bmap = tf.io.parse_tensor(example['bmap'], out_type=tf.uint8)

    image_shape = [example['height'], example['width'], example['depth']]

    image = tf.reshape(image, image_shape)
    label = tf.reshape(label, image_shape)
    bmap = tf.reshape(bmap, (image_shape[0], image_shape[1], 1))

    #imaged = tf.math.divide(tf.cast(image, dtype=tf.float32), 255)
    bmapd = tf.math.divide(tf.cast(bmap, dtype=tf.float32), 12)
    #labeld = tf.math.divide(tf.cast(label, dtype=tf.float32), 255)

    return image, label, bmapd


def read_tfrecord_new(serialized_example):
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

    image = tf.reshape(image, image_shape)
    label = tf.reshape(label, image_shape)
    bmap1 = tf.reshape(bmap1, (image_shape[0], image_shape[1], 1))
    bmap2 = tf.reshape(bmap2, (image_shape[0], image_shape[1], 1))

    # imaged = tf.math.divide(tf.cast(image, dtype=tf.float32), 255)
    # bmapd = tf.math.divide(tf.cast(bmap, dtype=tf.float32), 12)
    # labeld = tf.math.divide(tf.cast(label, dtype=tf.float32), 255)

    return image, label, bmap1, bmap2


def convert_to_index(sig_image):
    '''
    Due to the fact that the produced blur map (by Laplacian or DomainTr)
    has values different than [0.5,0.75,1,1.25,1.5,1.75,...,...,6], it has
    to be converted to this scale.
    :param sig_image:
    :return:
    '''
    k = np.arange(0.75, 6.25, 0.25)

    H, W = sig_image.shape
    sig_image_indexed = np.zeros(shape=(H,W,1), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            if sig_image[i,j] <= 0.5:
                sig_image_indexed[i,j,0] = 0
            else :

                pixel_std = np.argmin(np.abs(k - sig_image[i,j] ))
                bval = k[pixel_std]

                sig_image_indexed[i,j,0] = bval * 12

    return sig_image_indexed


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, label, bmap1, bmap2, image_shape):
    feature = {
        'image': _bytes_feature(image),
        'label': _bytes_feature(label),
        'bmap1': _bytes_feature(bmap1),
        'bmap2': _bytes_feature(bmap2),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
    }

    #  Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def produce_dataset():

    tfrecord_dir1 = 'tfrecords/data_train.tfrecords'
    tfrecord_dir2 = 'tfrecords/data_validation.tfrecords'
    BS = 8


    # Produce dataset for training
    tfrecord_dataset1 = tf.data.TFRecordDataset(tfrecord_dir1).shuffle(1024, seed=12)
    parsed_dataset1 = tfrecord_dataset1.map(read_tfrecord)
    parsed_dataset_train = parsed_dataset1.batch(BS)

    tfrecord_dir = 'tfrecords/dataset_train.tfrecords'
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:

        for i, single_batch in enumerate(parsed_dataset_train):
            batch_start = time.time()

            x = single_batch[0]
            y = single_batch[1]
            psf = single_batch[2]

            img = x.numpy()
            image_count = img.shape[0]
            bmap = psf.numpy()
            img_hat = y.numpy()
            _,h,w,c = img.shape

            count = image_count if BS > image_count else BS
            laplacian_nos = np.random.permutation(count)

            for k in range(count):
                no = laplacian_nos[k]
                if k < count / 2:
                    # propagate the ground truth blur values using Laplacian
                    bmapLaplacian = propagate_laplacian(img[no, :, :, :], bmap[no, :, :, 0])
                    bmap_indexed = convert_to_index(bmapLaplacian)
                else:
                    # propagate the ground truth blur values using DomainTr
                    bmapDomainTr = propagate_domaintransform(img[no, :, :, :], bmap[no, :, :, 0])
                    bmap_indexed = convert_to_index(bmapDomainTr)

                img_bytes = tf.io.serialize_tensor(img[no, :, :, :].reshape((h, w, c)))
                label_bytes = tf.io.serialize_tensor(img_hat[no, :, :, :].reshape((h, w, c)))
                psf_bytes1 = tf.io.serialize_tensor(bmap_indexed.reshape((h, w, 1)))
                psf_bytes2 = tf.io.serialize_tensor(np.uint8(bmap[no, :, :, :].reshape((h, w, 1)) * 12))

                example = serialize_example(img_bytes, label_bytes, psf_bytes1, psf_bytes2, (h,w,c))
                writer.write(example)

            batch_end = time.time()
            elapsed = (batch_end - batch_start) / 60.0

            print('train batch : {:d} - {:.4}'.format(i,elapsed))

        writer.close()


    # Produce dataset for validation
    tfrecord_dataset2 = tf.data.TFRecordDataset(tfrecord_dir2).shuffle(1024, seed=12)
    parsed_dataset2 = tfrecord_dataset2.map(read_tfrecord)
    parsed_dataset_test = parsed_dataset2.batch(BS)
    
    tfrecord_dir = 'tfrecords/dataset_validation.tfrecords'
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        for i, single_batch in enumerate(parsed_dataset_test):
            batch_start = time.time()
    
            x = single_batch[0]
            y = single_batch[1]
            psf = single_batch[2]
    
            img = x.numpy()
            image_count = img.shape[0]
            bmap = psf.numpy()
            img_hat = y.numpy()
            _,h,w,c = img.shape
    
            count = image_count if BS > image_count else BS
            laplacian_nos = np.random.permutation(count)
    
            for k in range(count):
                no = laplacian_nos[k]
                if k < count / 2:
                    bmapLaplacian = propagate_laplacian(img_hat[no, :, :, :], bmap[no, :, :, 0])
                    bmap_indexed = convert_to_index(bmapLaplacian)
                else:
                    bmapDomainTr = propagate_domaintransform(img_hat[no, :, :, :], bmap[no, :, :, 0])
                    bmap_indexed = convert_to_index(bmapDomainTr)
    
                img_bytes = tf.io.serialize_tensor(img[no, :, :, :].reshape(h,w,c))
                label_bytes = tf.io.serialize_tensor(img_hat[no, :, :, :].reshape(h,w,c))
                psf_bytes1 = tf.io.serialize_tensor(bmap_indexed.reshape(h,w, 1))
                psf_bytes2 = tf.io.serialize_tensor(np.uint8(bmap[no, :, :, :].reshape(h,w,1)))
    
                example = serialize_example(img_bytes, label_bytes, psf_bytes1, psf_bytes2, (h,w,c))
                writer.write(example)
    
            batch_end = time.time()
            elapsed = (batch_end - batch_start) / 60.0
    
            print('validation batch : {:d} - {:.4}'.format(i,elapsed))
    
        writer.close()


if __name__ == '__main__':
    produce_dataset()