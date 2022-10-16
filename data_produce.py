import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import cv2
import time

from  disk_kernels import disk_kernels_mat as disk_kernels



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


def serialize_example(image, label, bmap, image_shape):
    feature = {
        'image': _bytes_feature(image),
        'label': _bytes_feature(label),
        'bmap': _bytes_feature(bmap),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
    }

    #  Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


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

    return image, label, bmap


def filter_with_PSF(image, sig_image, disk_kernels):
    H,W,c = image.shape
    padsize = 21
    dst = cv2.copyMakeBorder(image, padsize, padsize, padsize, padsize, cv2.BORDER_REPLICATE)
    dst = dst.astype(np.float32)/255.

    blurred_image = np.zeros((H,W,c), dtype=np.uint8)
    k = np.arange(0.75, 6.25, 0.25)

    sig_image_rounded = np.zeros((H, W, 1), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            pixel_std = sig_image[i,j]

            if pixel_std != 0:
                pixel_std = np.argmin(np.abs(k - pixel_std))
                sig_image_rounded[i, j] = k[pixel_std]
                disk_kernel = disk_kernels[pixel_std]

                ii = padsize + i
                jj = padsize + j

                sizedisk, _ = disk_kernel.shape
                sizedisk = np.floor(sizedisk/2).astype(np.uint8)

                cropB = dst[ii-sizedisk:ii+sizedisk+1, jj-sizedisk:jj+sizedisk+1,0]
                cropG = dst[ii-sizedisk:ii+sizedisk+1, jj-sizedisk:jj+sizedisk+1,1]
                cropR = dst[ii-sizedisk:ii+sizedisk+1, jj-sizedisk:jj+sizedisk+1,2]

                cB = np.sum(np.multiply(cropB, disk_kernel))
                cG = np.sum(np.multiply(cropG, disk_kernel))
                cR = np.sum(np.multiply(cropR, disk_kernel))

                blurred_image[i,j,0] = (cB*255).astype(np.uint8)
                blurred_image[i,j,1] = (cG*255).astype(np.uint8)
                blurred_image[i,j,2] = (cR*255).astype(np.uint8)
            else:

                blurred_image[i,j,0] = image[i,j,0]
                blurred_image[i,j,1] = image[i,j,1]
                blurred_image[i,j,2] = image[i,j,2]

    return blurred_image, sig_image_rounded


def produce_PSF1(H, W):

    # from left to right increasing
    k = np.arange(0.75, 6.25, 0.25)
    leftstart_max = 100
    sig_image = np.zeros(shape=(3, H, W), dtype=np.float32)

    for btype in range(3):
        start_end_coord = np.random.randint(low=0, high=leftstart_max, size=2)
        startc = start_end_coord[0]
        endc = start_end_coord[1] + (W-leftstart_max)

        start_end_blur = np.random.randint(low=0, high=6, size=2)
        startb = k[start_end_blur[0]]
        endb = k[start_end_blur[1] + 16]
        nline = endc - startc + 1

        blur_line = np.linspace(startb, endb, num=nline)
        line = np.zeros(shape=(1,W), dtype=np.float32)
        line[0, endc:] = endb

        line[0,startc:endc+1] = blur_line
        sig_image[btype, :, :] = line

        # bimage[btype, :,:,:] = filter_with_PSF(newimg, sig_image, disk_kernels)
    #     plt.matshow(sig_image)
    #
    # plt.show()

    return sig_image


def produce_PSF2(H, W):
    # from up to down increasing

    k = np.arange(0.75, 6.25, 0.25)
    upstart_max = 100
    sig_image = np.zeros(shape=(3, H, W), dtype=np.float32)

    for btype in range(3):
        start_end_coord = np.random.randint(low=0, high=upstart_max, size=2)
        startc = start_end_coord[0]
        endc = start_end_coord[1] + (H-upstart_max)

        start_end_blur = np.random.randint(low=0, high=6, size=2)
        startb = k[start_end_blur[0]]
        endb = k[start_end_blur[1] + 16]
        nline = endc - startc + 1

        blur_line = np.linspace(startb, endb, num=nline)
        line = np.zeros(shape=(H,1), dtype=np.float32)
        line[endc:, 0] = endb

        line[startc:endc+1, 0] = blur_line
        sig_image[btype, :,:] = line
    #
    #     bimage[btype, :,:,:] = filter_with_PSF(newimg, sig_image, disk_kernels)
    #     plt.matshow(sig_image)
    #
    # plt.show()

    return sig_image


def produce_PSF3(H, W):
    # from left to right decreasing

    k = np.arange(0.75, 6.25, 0.25)
    leftstart_max = 100
    sig_image = np.zeros(shape=(3, H, W), dtype=np.float32)

    for btype in range(3):
        start_end_coord = np.random.randint(low=0, high=100, size=2)
        startc = start_end_coord[0]
        endc = start_end_coord[1] + (W-leftstart_max)

        start_end_blur = np.random.randint(low=0, high=6, size=2)
        startb = k[start_end_blur[0]]
        endb = k[start_end_blur[1] + 16]
        nline = endc - startc + 1

        blur_line = np.linspace(startb, endb, num=nline)
        line = np.zeros(shape=(W,), dtype=np.float32)
        line[endc:] = endb

        line[startc:endc+1] = blur_line
        newline = np.flip(line,0)

        sig_image[btype, :,:] = newline

    return sig_image


def produce_PSF4(H, W):

    # from up to down decreasing
    k = np.arange(0.75, 6.25, 0.25)
    upstart_max = 100
    sig_image = np.zeros(shape=(3, H, W), dtype=np.float32)

    for btype in range(3):
        start_end_coord = np.random.randint(low=0, high=upstart_max, size=2)
        startc = start_end_coord[0]
        endc = start_end_coord[1] + (H-upstart_max)

        start_end_blur = np.random.randint(low=0, high=6, size=2)
        startb = k[start_end_blur[0]]
        endb = k[start_end_blur[1] + 16]
        nline = endc - startc + 1

        blur_line = np.linspace(startb, endb, num=nline)
        line = np.zeros(shape=(H,), dtype=np.float32)
        line[endc:] = endb

        line[startc:endc+1] = blur_line

        line2 = np.flip(line, 0)
        sig_image[btype, :,:] = line2[:,np.newaxis]

    return sig_image


def produce_PSF5(H, W):

    # from left to right step-wise increase
    k = np.arange(0.75, 6.25, 0.25)
    sig_image = np.zeros(shape=(3, H, W), dtype=np.float32)
    leftstart_max = 141

    for btype in range(3):
        start_coord = np.random.randint(low=0, high=leftstart_max, size=1)
        startc = start_coord[0]
        start_blur = np.random.randint(low=0, high=15, size=1)

        line = np.zeros(shape=(1, W), dtype=np.float32)
        line[0, startc+0   : startc + 50] = k[start_blur[0]]
        line[0, startc+50  : startc + 100] = k[start_blur[0]+2]
        line[0, startc+100 : startc + 150] = k[start_blur[0]+4]
        line[0, startc+150 : startc + 200] = k[start_blur[0]+6]
        line[0, startc + 199:] = k[start_blur[0]+6]

        sig_image[btype, :,:] = line

    return sig_image


def produce_PSF6(H, W):

    # from left to right step-wise decreasing
    k = np.arange(0.75, 6.25, 0.25)
    sig_image = np.zeros(shape=(3, H, W), dtype=np.float32)
    leftstart_max = 141

    for btype in range(3):
        start_coord = np.random.randint(low=0, high=leftstart_max, size=1)
        startc = start_coord[0]
        start_blur = np.random.randint(low=0, high=15, size=1)

        line = np.zeros(shape=(1, W), dtype=np.float32)
        line[0, :startc] = k[start_blur[0]+6]
        line[0, startc+0   : startc + 50] = k[start_blur[0]+6]
        line[0, startc+50  : startc + 100] = k[start_blur[0]+4]
        line[0, startc+100 : startc + 150] = k[start_blur[0]+2]
        line[0, startc+150 : startc + 200] = k[start_blur[0]]
        line[0, startc + 199:] = k[start_blur[0]]

        sig_image[btype, :,:] = line

    return sig_image


def produce_PSF7(H, W):

    # from up to down step-wise increase
    k = np.arange(0.75, 6.25, 0.25)
    sig_image = np.zeros(shape=(3, H, W), dtype=np.float32)
    leftstart_max = 141

    for btype in range(3):
        start_coord = np.random.randint(low=0, high=leftstart_max, size=1)
        startc = start_coord[0]
        start_blur = np.random.randint(low=0, high=15, size=1)

        line = np.zeros(shape=(H, 1), dtype=np.float32)
        line[startc+0   : startc + 50,  0] = k[start_blur[0]]
        line[startc+50  : startc + 100, 0] = k[start_blur[0]+2]
        line[startc+100 : startc + 150, 0] = k[start_blur[0]+4]
        line[startc+150 : startc + 200, 0] = k[start_blur[0]+6]
        line[startc + 199:, 0] = k[start_blur[0]+6]

        sig_image[btype, :,:] = line

    return sig_image


def produce_PSF8(H, W):

    # from up to down step-wise decreasing
    k = np.arange(0.75, 6.25, 0.25)
    sig_image = np.zeros(shape=(3, H, W), dtype=np.float32)
    leftstart_max = 141

    for btype in range(3):
        start_coord = np.random.randint(low=0, high=leftstart_max, size=1)
        startc = start_coord[0]
        start_blur = np.random.randint(low=0, high=15, size=1)

        line = np.zeros(shape=(H, 1), dtype=np.float32)
        line[:startc, 0] = k[start_blur[0]+6]
        line[startc+0   : startc + 50, 0] = k[start_blur[0]+6]
        line[startc+50  : startc + 100, 0] = k[start_blur[0]+4]
        line[startc+100 : startc + 150, 0] = k[start_blur[0]+2]
        line[startc+150 : startc + 200, 0] = k[start_blur[0]]
        line[startc + 199:, 0] = k[start_blur[0]]

        sig_image[btype, :,:] = line

    return sig_image


def produce_PSF9(H, W):
    # uniform blur
    k = np.arange(0.75, 6.25, 0.25)
    sig_image = np.zeros(shape=(3, H, W), dtype=np.float32)
    for btype in range(3):
        start_blur = np.random.randint(low=0, high=22, size=1)
        sig_image[btype, :, :] = k[start_blur[0]]

    return sig_image


def produce_PSF10(H,W):
    sig_images1 = produce_PSF1(H, W)
    sig_images2 = produce_PSF2(H, W)

    sig_image = np.sqrt(np.multiply(sig_images1, sig_images2))

    return sig_image


def produce_PSF11(H, W):
    sig_images1 = produce_PSF3(H, W)
    sig_images2 = produce_PSF4(H, W)

    sig_image = np.sqrt(np.multiply(sig_images1, sig_images2))

    return sig_image


def produce_PSF12(H, W):
    sig_images1 = produce_PSF1(H, W)
    sig_images2 = produce_PSF4(H, W)

    sig_image = np.sqrt(np.multiply(sig_images1, sig_images2))

    return sig_image


def produce_PSF13(H, W):
    sig_images1 = produce_PSF2(H, W)
    sig_images2 = produce_PSF3(H, W)

    sig_image = np.sqrt(np.multiply(sig_images1, sig_images2))

    return sig_image


def convert_to_index(sig_image):
    k = np.arange(0.75, 6.25, 0.25)

    H, W = sig_image.shape
    sig_image_indexed = np.zeros(shape=(H,W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            if sig_image[i,j] != 0.0:

                pixel_std = np.argmin(np.abs(k - sig_image[i,j] ))
                bval = k[pixel_std]

                tval = np.argwhere(k == bval)

                sig_image_indexed[i,j] = (tval[0][0] + 1) * 12

    return sig_image_indexed


def produce_data():

    data_dir1 = 'selected_IMAGENET/'
    data_dir2 = 'selected_COCO/'
    image_paths1 = glob.glob(data_dir1 + '*.JPEG')
    image_paths2 = glob.glob(data_dir2 + '*.jpg')
    image_paths_train = image_paths1[:280] + image_paths2[:280]
    image_paths_validation = image_paths1[280:] + image_paths2[280:]


    
    tfrecord_dir = 'tfrecords/data_train.tfrecords'
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        kk = 1
        for image_path in image_paths_train:
            start_time1 = time.time()
    
            img = cv2.imread(image_path)
    
            H = 360
            W = 340
    
            newimg = cv2.resize(img, (W, H))
    
            sig_images = np.zeros(shape=(13*3, H,W), dtype=np.float32)
    
            sig_images[:3,:,:] = produce_PSF1(H, W)
            sig_images[3:6,:,:] = produce_PSF2(H, W)
            sig_images[6:9,:,:] = produce_PSF3(H, W)
            sig_images[9:12,:,:] = produce_PSF4(H, W)
            sig_images[12:15,:,:] = produce_PSF5(H, W)
            sig_images[15:18,:,:] = produce_PSF6(H, W)
            sig_images[18:21,:,:] = produce_PSF7(H, W)
            sig_images[21:24,:,:] = produce_PSF8(H, W)
            sig_images[24:27,:,:] = produce_PSF9(H, W)
            sig_images[27:30,:,:] = produce_PSF10(H, W)
            sig_images[30:33,:,:] = produce_PSF11(H, W)
            sig_images[33:36,:,:] = produce_PSF12(H, W)
            sig_images[36:39,:,:] = produce_PSF13(H, W)
    
    
            for i in range(13*3):
                bimage, psfimage_rounded = filter_with_PSF(newimg, sig_images[i, :, :], disk_kernels)
                image_shape = bimage.shape
                psfimage = np.uint8(psfimage_rounded * 12)
    
                img_bytes = tf.io.serialize_tensor(bimage)
                label_bytes = tf.io.serialize_tensor(newimg)
                psf_bytes = tf.io.serialize_tensor(psfimage)
    
                example = serialize_example(img_bytes, label_bytes, psf_bytes, image_shape)
                writer.write(example)
    
            elapsed_time1 = time.time() - start_time1
            time.strftime("%H:%M:%S", time.gmtime(elapsed_time1))
    
            print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time1)), end='')
            print(' TRAIN - {:d} \n'.format(kk))


    tfrecord_dir = 'tfrecords/data_validation.tfrecords'
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        kk = 1
        for image_path in image_paths_validation:
            start_time2 = time.time()

            img = cv2.imread(image_path)
            H1, W1, c = img.shape

            H = 360
            W = 340

            if H1 < H:
                newimg = cv2.resize(img, (W1, H))
            else:
                startci = np.random.randint(low=H, high=H1 + 1, size=1)
                newimg = img[startci[0] - H:startci[0], :, :]

            H1 = H

            if W1 < W:
                newimg = cv2.resize(newimg, (W, H1))
            else:
                startci = np.random.randint(low=W, high=W1 + 1, size=1)
                newimg = newimg[:, startci[0] - W:startci[0], :]

            sig_images = np.zeros(shape=(13 * 3, H, W), dtype=np.float32)
     
            sig_images[:3, :, :] = produce_PSF1(H, W)
            sig_images[3:6, :, :] = produce_PSF2(H, W)
            sig_images[6:9, :, :] = produce_PSF3(H, W)
            sig_images[9:12, :, :] = produce_PSF4(H, W)
            sig_images[12:15, :, :] = produce_PSF5(H, W)
            sig_images[15:18, :, :] = produce_PSF6(H, W)
            sig_images[18:21, :, :] = produce_PSF7(H, W)
            sig_images[21:24, :, :] = produce_PSF8(H, W)
            sig_images[24:27, :, :] = produce_PSF9(H, W)
            sig_images[27:30, :, :] = produce_PSF10(H, W)
            sig_images[30:33, :, :] = produce_PSF11(H, W)
            sig_images[33:36, :, :] = produce_PSF12(H, W)
            sig_images[36:39, :, :] = produce_PSF13(H, W)

            for i in range(13 * 3):
                bimage, psfimage_rounded = filter_with_PSF(newimg, sig_images[i, :, :], disk_kernels)
                image_shape = bimage.shape
                psfimage = np.uint8(psfimage_rounded * 12)

                img_bytes = tf.io.serialize_tensor(bimage)
                label_bytes = tf.io.serialize_tensor(newimg)
                psf_bytes = tf.io.serialize_tensor(psfimage)

                example = serialize_example(img_bytes, label_bytes, psf_bytes, image_shape)
                writer.write(example)

            elapsed_time2 = time.time() - start_time2
            time.strftime("%H:%M:%S", time.gmtime(elapsed_time2))

            print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time2)), end=' ')
            print('VALIDATION - {:d} \n'.format(kk))


if __name__ == '__main__':
    produce_data()