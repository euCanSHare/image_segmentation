import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.
aux_root = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.join(*os.path.split(aux_root)[:-1])

data_base = os.path.join(project_root, 'data')
log_root = os.path.join(project_root, 'models')
