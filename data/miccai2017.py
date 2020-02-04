import os
import glob

from data.generichandler import GenericHandler

wd = os.path.dirname(os.path.realpath(__file__))


class Miccai2017(GenericHandler):
    '''
    Class for handling data files from ACDC MICCAI 2017 dataset.
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

            infos = {}
            for line in open(os.path.join(folder_path, 'Info.cfg')):
                label, value = line.split(':')
                infos[label] = value.rstrip('\n').lstrip(' ')

            patient_id = folder.lstrip('patient')
            ED_frame = int(infos['ED'])

            for f in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                new_file = {'image': f, 'mask': '', 'info': infos.copy(), 'patient_id': patient_id}
                frame = int(f.rstrip('.nii.gz').split('frame')[-1])
                new_file['phase'] = 'ED' if frame == ED_frame else 'ES'

                maskf = f.rstrip('.nii.gz') + '_gt.nii.gz'
                if os.path.exists(maskf):
                    new_file['mask'] = maskf

                file_list.append(new_file)

        return file_list