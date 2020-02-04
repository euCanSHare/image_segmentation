import os
import glob

wd = os.path.dirname(os.path.realpath(__file__))


class GenericHandler(object):
    '''
    Class for handling generic data files.
    '''
    def __init__(self, mode='training'):
        '''Constructor.'''
        self.dataset   = self.__class__.__name__.lower() # Name of class in lowercase format.
        self.base_path = os.path.join(wd, self.dataset, mode)

    def get_files(self):
        '''Obtain all file paths for each of the corresponding sets (training and testing).'''
        return self.get_files_in_folder(self.base_path)

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

            for f in glob.glob(os.path.join(folder_path, '*.nii.gz')):

                new_file = {'image': f, 'mask': '', 'info': {}, 'patient_id': patient_id}
                new_file['phase'] = f.rstrip('.nii.gz')[-2:]
                sp = f.split('_sa_')
                maskf = sp[0] + '_seg_sa_' + sp[1]
                if os.path.exists(maskf):
                    new_file['mask'] = maskf

                file_list.append(new_file)

        return file_list