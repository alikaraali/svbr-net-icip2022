import tensorflow as tf
import cv2
import argparse
import numpy as np

from tensorflow.keras.optimizers import Adam
from train import *



def test_one_image(model, trained_weights_path, image, psf):
    """_summary_

    Args:
        model (_type_): _description_
        trained_weights_path (_type_): _description_
        image (_type_): _description_
        psf (_type_): _description_

    Returns:
        _type_: _description_
    """

    INIT_LR = 1e-3

    opt = Adam(learning_rate=INIT_LR)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), opt=opt, net=model)

    manager = tf.train.CheckpointManager(ckpt, trained_weights_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    image = np.expand_dims(image, axis=0)
    psf = np.expand_dims(psf, axis=0)

    return model.predict([image, psf])


def get_args():
    parser = argparse.ArgumentParser(description='SVBR_Net \n')

    parser.add_argument('-bimage_path', metavar='--blurry_image_path', required=True,
                        type=str, help='Path of the efocused image(s) \n')

    parser.add_argument('-bmap_path', metavar='--bmap_path', required=True,
                        type=str, help='Path of the corresponding blur map(s)\n')

    parser.add_argument('-p', metavar='--weights_path', required=False,
                        type=str, help='Path for the weights \n', 
                        default='training/')

    args = parser.parse_args()
    image = args.image_path
    bmap = args.bmap_path
    trained_wp = args.p

    return image, bmap, trained_wp


if __name__ == '__main__':

    image_path, bmap_path, trained_wp = get_args()

    image = cv2.imread(image_path)
    dimage = image / 255.0
    bmap = cv2.imread(bmap_path)
    bmap = bmap[:,:,0:1] if bmap.ndim == 3 else bmap
    dbmap = bmap / 255.0

    model = make_model1()
    pred = test_one_image(model, trained_wp, dimage, dbmap)
    deconvolved_im = tf.cast(tf.math.multiply(pred[0, :, :, :], 255), dtype=tf.uint8).numpy()
    cv2.imwrite(image_path + 'deconv.png', deconvolved_im)
