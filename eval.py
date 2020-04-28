import os
import glob
import logging
from importlib import machinery

from model import Model
from evaluate_patients import score_data as evaluation


def run(model_path, datasets, output_path):

    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = machinery.SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    # output_path = os.path.dirname(datasets[0])
    logging.warning('Saving segmentations on {}'.format(output_path))

    model = Model(exp_config)

    init_iteration = evaluation(model, output_path, model_path,
                                datasets, exp_config=exp_config,
                                do_postprocessing=True)

    return init_iteration
