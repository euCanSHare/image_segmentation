import os
import re
import glob
import time
import logging
import numpy as np
import tensorflow as tf


from skimage import transform

from utils import utils_gen, utils_nii, image_utils
from data.dataset import Dataset



def crop_or_pad_volume_to_size(vol, nx, ny):

    x, y, z = vol.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        vol_cropped = vol[x_s:x_s + nx, y_s:y_s + ny, :]
    else:
        vol_cropped = np.zeros((nx, ny, z))
        if x <= nx and y > ny:
            vol_cropped[x_c:x_c + x, :, :] = vol[:, y_s:y_s + ny, :]
        elif x > nx and y <= ny:
            vol_cropped[:, y_c:y_c + y, :] = vol[x_s:x_s + nx, :, :]
        else:
            vol_cropped[x_c:x_c + x, y_c:y_c + y, :] = vol[:, :, :]

    return vol_cropped


def score_data(model, output_folder, model_path, datasets, exp_config, do_postprocessing=False):

    batch_size = 1
    num_channels = exp_config.nlabels

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Backward compatibility
    if 'model_type' not in exp_config.__dict__.keys():
        exp_config.model_type = 'convolutional'

    with tf.Session() as sess:

        sess.run(init)
        checkpoint_path = utils_gen.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        init_iteration = int(checkpoint_path.split('/')[-1].split('-')[-1])

        total_time = 0
        total_volumes = 0

        # Select image pixel size
        nx, ny = exp_config.image_size[:2]
        data = Dataset(datasets, exp_config.data_mode, (nx, ny), exp_config.target_resolution)

        # Iterate over volumes and slices of dataset
        for idx, volume in enumerate(data):
            predictions = []

            logging.info(' ----- Doing image: -------------------------')
            logging.info(' {}'.format(volume.filepath))
            logging.info(' --------------------------------------------')

            img_dat = []
            start_time = time.time()

            for i, slc in enumerate(volume):
                if i == 0:
                    img_dat = volume.img_dat

                x, y = slc.shape
                slice_cropped = slc.img_cropped
                x_s, y_s, x_c, y_c = slc.cropped_boundaries

                # GET PREDICTION
                network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
                _, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
                prediction_cropped = np.squeeze(logits_out[0,...])

                # ASSEMBLE BACK THE SLICES
                slice_predictions = np.zeros((x,y,num_channels))
                # insert cropped region into original image again
                if x > nx and y > ny:
                    slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = prediction_cropped
                else:
                    if x <= nx and y > ny:
                        slice_predictions[:, y_s:y_s+ny,:] = prediction_cropped[x_c:x_c+ x, :,:]
                    elif x > nx and y <= ny:
                        slice_predictions[x_s:x_s + nx, :,:] = prediction_cropped[:, y_c:y_c + y,:]
                    else:
                        slice_predictions[:, :,:] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y,:]

                # RESCALING ON THE LOGITS
                prediction = transform.resize(slice_predictions,
                                              (img_dat[0].shape[0], img_dat[0].shape[1], num_channels),
                                              order=1,
                                              preserve_range=True,
                                              mode='constant')

                prediction = np.uint8(np.argmax(prediction, axis=-1))
                predictions.append(prediction)

            prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

            # This is the same for 2D and 3D again
            if do_postprocessing:
                prediction_arr = image_utils.keep_largest_connected_components(prediction_arr)

            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_volumes += 1

            logging.info('Evaluation of volume took {0} secs.'.format(elapsed_time))

            # Save prediced mask
            out_file_name = volume.filepath.rstrip('.nii.gz') + '_mask.nii.gz'
            out_affine = img_dat[1]
            out_header = img_dat[2]

            logging.info('saving to: {}'.format(out_file_name))
            utils_nii.save_nii(out_file_name, prediction_arr, out_affine, out_header)

        logging.info('Average time per volume: {}'.format(total_time/total_volumes))

    return init_iteration
