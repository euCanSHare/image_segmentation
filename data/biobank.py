import os
import re
import glob

from data.generichandler import GenericHandler

wd = os.path.dirname(os.path.realpath(__file__))


class Biobank(GenericHandler):
    '''
    Class for handling data files from Biobank dataset.
    '''
    def get_files_in_folder(self, folder):
        '''
        Process all files inside given folder and extract available information.
        Returns:
            returns a list of files with keys info, image, mask, phase, patient_id
        '''
        file_list = []
        for folder in sorted(os.listdir(self.base_path)):
            folder_path = os.path.join(self.base_path, folder)

            patient_id = folder

            for f in glob.glob(os.path.join(folder_path, '???????_sa_frame??_E?.nii.gz')):

                new_file = {'image': f, 'mask': '', 'info': {}, 'patient_id': patient_id}
                new_file['phase'] = f.rstrip('.nii.gz')[-2:]
                sp = f.split('_sa_')
                maskf = sp[0] + '_seg_sa_' + sp[1]
                if os.path.exists(maskf):
                    new_file['mask'] = maskf

                file_list.append(new_file)

        return file_list