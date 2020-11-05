import os
import re
import glob
import time
import logging
import numpy as np
import nibabel as nib
import tensorflow as tf

from skimage import transform

from utils import utils_gen, utils_nii, image_utils
from data.dataset import Dataset

LABELS = {
    'lv': 3, 'myo': 2, 'rv': 1
}

def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2
    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


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


def crop_or_pad_4D(tensor4d, nx, ny, nz, nt):
    tensor4d = np.array(tensor4d)
    _, _, z, t = tensor4d.shape
    ph = np.zeros([nx, ny, nz, nt])
    if nz > z:
        for frame in range(0, t, int(t/nt)):
            for slc in range(z):
                # try:
                ph[:,:,slc + int((nz-z)/2),int(frame/int(t/nt))] = crop_or_pad_slice_to_size(tensor4d[:,:,slc,frame],nx,ny)
                # except:
                #     a = 1
    else:
        for frame in range(0, t, int(t/nt)):
            for slc in range(nz):
                # try:
                ph[:,:,slc,int(frame/int(t/nt))] = crop_or_pad_slice_to_size(tensor4d[:,:,slc + int((z-nz)/2),frame],nx,ny)
                # except:
                #     a = 1
    return ph


def crop_or_pad(tensor, nx, ny, nz, nt):
    if tensor.ndim == 4:
        res = crop_or_pad_4D(tensor, nx, ny, nz, nt)
    elif tensor.ndim == 3:
        res = crop_or_pad_volume_to_size(tensor, nx, ny)
    else:
        raise ValueError('Shape of input dataset is invalid "{}". It must be 3 or 4'.format(tensor.ndim))

    return res

def normalise_image(image, mean=0, std=1):
    '''
    make image zero mean and unit standard deviation (default values)
    '''
    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide(std*(img_o - m + mean), s)


def cine_2_tensor_lst(cine):    
    if cine.ndim == 3:
        cine = np.expand_dims(cine, axis=-1)

    tensor = []
    for t in range(cine.shape[3]):
        for z in range(cine.shape[2]):
            tensor.append(
                np.asarray(
                    np.expand_dims(
                        np.expand_dims(
                            normalise_image(cine[...,z,t],0.5,0.5),axis = 0
                            ),axis = 3
                        ),np.float32)
                    )
            
    return tensor


def tensor_lst_2_cine(tensor_lst, z):
    if len(tensor_lst) == 1:
        return np.squeeze(tensor_lst[0])

    cine = []
    for idx,_ in enumerate(tensor_lst):

        if idx%z == 0:
            time_seq = []
            for time in range(z):   
                time_seq.append(np.squeeze(tensor_lst[idx+time]))

            cine.append(time_seq) 

    cine = np.moveaxis(np.array(cine), [0,1], [3,2])

    return cine   


def score_data(model, output_folder, model_path, datasets, exp_config, do_postprocessing=False):

    batch_size = 1

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Backward compatibility
    if 'model_type' not in exp_config.__dict__.keys():
        exp_config.model_type = 'convolutional'

    output_files = []
    with tf.Session() as sess:

        sess.run(init)
        checkpoint_path = utils_gen.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        # Select image pixel size
        nx, ny = exp_config.image_size[:2]

        for _file in datasets:
            print('-'*20)
            print(' Segmenting file "{}"'.format(_file))
            print('-'*20)
            try:
                nim = nib.load(_file)
                header = nim.header
                data = nim.get_fdata()
                data = crop_or_pad(data, nx, ny, header['dim'][3], header['dim'][4])
            except:
                logging.info('Unable to read: {0}'.format(_file))

            x_lst = cine_2_tensor_lst(data)
            
            msks_out = []
            for x in x_lst:

                feed_dict = {images_pl: x}
    
                mask_out, _ = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)
                msks_out.append(mask_out)
            
            cine = tensor_lst_2_cine(msks_out, header['dim'][3])
            reshaped = crop_or_pad(cine, header['dim'][1], header['dim'][2], header['dim'][3], header['dim'][4])   
            means = nib.Nifti1Image(reshaped, header=header, affine=np.eye(4))  

            save_name = _file.split('.')[0] + '_label.nii.gz'
            print('Saving ', save_name)
            nib.save(means, save_name)

            metadata = {
                'labels': LABELS,
                'format': '3D'
            }

            if reshaped.shape[-1] > 1:
                # Compute ED and ES positions
                aux = reshaped.copy()
                aux[aux <= 2] = 0
                aux[aux > 0] = 1
                volume = np.sum(aux, axis=(0,1,2))
                ed, es = np.argmax(volume), np.argmin(volume)

                metadata.update({
                    'ED': int(ed), 'ES': int(es)
                })
                metadata['format'] = '4D'

            output_files.append((save_name, metadata))

    return output_files
