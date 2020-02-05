import os
import sys
import glob
import logging
import argparse
from importlib.machinery import SourceFileLoader

import config.system as sys_config
from utils import utils_gen
from model import Model
from evaluate_patients import score_data as evaluation



wd = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description="Script to evaluate a neural network model"
)
parser.add_argument("MODEL_NAME", type=str, help="Name of experiment to use")
parser.add_argument('-i', '--iter', type=int, help='which iteration to use')
parser.add_argument('-d', '--dataset', type=str, help='Select which dataset to evaluate.')


if __name__ == "__main__":
    args = parser.parse_args()

    dataset = args.dataset
    assert dataset is not None, 'Please, provide dataset to segment.'
    all_datasets = [f for f in next(os.walk(sys_config.data_base))[1] if f[:2] != '__']
    if '-' in dataset:
        datasets = dataset.split('-')
        logging.info('Evaluating more than one dataset at the same time: {0}'.format(datasets))
        for ds in datasets:
            assert ds in all_datasets, 'Dataset "{0}" not available. \
                Please, choose one of the following: {1}'.format(ds, all_datasets)
    else:
        assert dataset in all_datasets, 'Dataset "{0}" not available. \
            Please, choose one of the following: {1}'.format(dataset, all_datasets)


    use_iter = args.iter
    if use_iter:
        logging.info('Using iteration: {0}'.format(use_iter))

    model_path = os.path.join(sys_config.log_root, args.MODEL_NAME)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    output_path = os.path.join(model_path, dataset, 'predictions')
    logging.warning('Saving segmentations on {}'.format(output_path))

    model = Model(exp_config)
    database = dataset if '-' not in dataset else datasets

    for outp, subset in zip([output_path], ['training', 'testing']):
        path_pred = os.path.join(outp, 'prediction')
        path_image = os.path.join(outp, 'image')
        utils_gen.makefolder(path_pred)
        utils_gen.makefolder(path_image)

        path_gt = os.path.join(outp, 'ground_truth')
        path_diff = os.path.join(outp, 'difference')
        path_eval = os.path.join(outp, 'eval')
        utils_gen.makefolder(path_diff)
        utils_gen.makefolder(path_gt)

        init_iteration = evaluation(model,
                                    outp,
                                    model_path,
                                    database,
                                    subset,
                                    exp_config=exp_config,
                                    do_postprocessing=True,
                                    evaluate_all=True,
                                    use_iter=use_iter)
