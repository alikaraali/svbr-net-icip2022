import tensorflow as tf
import sys
import time
import argparse

from models import make_model1
from tensorflow.keras.optimizers import Adam
from utils import read_tfrecord_new


def SSIMLoss(y_true, y_pred):
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def step(model, opt, x, psf, y):
    """_summary_

    Args:
        model (_type_): _description_
        opt (_type_): _description_
        x (_type_): _description_
        psf (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    # keep track of our gradients
    with tf.GradientTape() as tape:
        pred = model.call([x, psf])
        loss = SSIMLoss(y, pred)


    # calculate the gradients using our tape and then update the
    # model weights
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train_model(model, opt, tfrecord_dir1, tfrecord_dir2, ckpoint_file, 
                    initial_lr, EPOCHS1, EPOCHS2, BS):
    """_summary_

    Args:
        model (_type_): _description_
        opt (_type_): _description_
        tfrecord_dir1 (_type_): _description_
        tfrecord_dir2 (_type_): _description_
        ckpoint_file (_type_): _description_
        initial_lr (_type_): _description_
        EPOCHS1 (_type_): _description_
        EPOCHS2 (_type_): _description_
        BS (_type_): _description_
    """

    tfrecord_dataset1 = tf.data.TFRecordDataset(tfrecord_dir1).shuffle(1024, seed=12)
    parsed_dataset1 = tfrecord_dataset1.map(read_tfrecord_new)
    parsed_dataset_train = parsed_dataset1.batch(BS)

    tfrecord_dataset2 = tf.data.TFRecordDataset(tfrecord_dir2).shuffle(1024, seed=12)
    parsed_dataset2 = tfrecord_dataset2.map(read_tfrecord_new)
    parsed_dataset_validation = parsed_dataset2.batch(BS)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), opt=opt, net=model)
    manager = tf.train.CheckpointManager(ckpt, ckpoint_file, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    opt.learning_rate = initial_lr

    # loop over the number of epochs
    for epoch in range(EPOCHS1 + EPOCHS2):
        # show the current epoch number
        sys.stdout.flush()
        epochStart = time.time()

        kk1 = 0
        loss_total = 0
        for i, single_batch in enumerate(parsed_dataset_train):
            x = single_batch[0]
            y = single_batch[1]
            psf = single_batch[3] if epoch < EPOCHS1 else single_batch[2]
 
            loss_batch = step(model, opt, x, psf, y)
            loss_total += loss_batch

            kk1 += 1

        ckpt.step.assign_add(1)
        save_path = manager.save()

        ## validation part
        sys.stdout.flush()
        kk2 = 0
        loss_total_val = 0
        for i, single_batch in enumerate(parsed_dataset_validation):
            x = single_batch[0]
            y = single_batch[1]
            psf = single_batch[3] if epoch < EPOCHS1 else single_batch[2]

            pred = model.call([x, psf])
            loss_batch = SSIMLoss(y, pred)
            loss_total_val += loss_batch

            kk2 += 1

        # Epoch decay. Change every 20 epoch.
        if epoch % 20 == 0:
            opt.learning_rate = opt.learning_rate.numpy() * 0.1

        # show timing information for the epoch
        epochEnd = time.time()
        elapsed = (epochEnd - epochStart) / 60.0

        print(f'Epoch : {epoch}, lr : {opt.learning_rate.numpy():.6f},', end=' ')
        print(f'Train loss : {(loss_total / kk1):.5f},', end=' ') 
        print(f'Val loss : {(loss_total_val / kk2):.5f},', end=' ')
        print(f'Time Passed : {elapsed:.2f}')
               

def train_network(tfrecord_dir1, tfrecord_dir2, ckpoint_file, INIT_LR, BS):
    """_summary_

    Args:
        tfrecord_dir1 (_type_): _description_
        tfrecord_dir2 (_type_): _description_
        ckpoint_file (_type_): _description_
        INIT_LR (_type_): _description_
        BS (_type_): _description_
    """
    opt = Adam(learning_rate=INIT_LR)

    model = make_model1()
    train_model(model, opt, tfrecord_dir1, tfrecord_dir2, ckpoint_file, INIT_LR, 2, 2, BS)


def get_args():
    parser = argparse.ArgumentParser(description='SVBR - A NON-BLIND SPATIALLY' + 
                                                'VARYING DEFOCUS BLUR REMOVAL NETWORK\n')

    parser.add_argument('-t', metavar='--train_data', required=False,
                        type=str, help='TF.Record file to train \n', 
                        default='tfrecords/dataset_train.tfrecords')

    parser.add_argument('-v', metavar='--validation_data', required=False,
                        type=str, help='TF.Record file to validate \n', 
                        default='tfrecords/dataset_validation.tfrecords')

    parser.add_argument('-c', metavar='--checkpoint_folder', required=False,
                        type=str, help='A path to save the checkpoint files \n', 
                        default='training/')

    args = parser.parse_args()

    return args.t, args.v, args.c



if __name__ == '__main__':

    BS = 12
    INIT_LR = 1e-3
    tf_train, tf_val, chekpoint_path = get_args()
    
    train_network(tf_train, tf_val, chekpoint_path, INIT_LR, BS)
