import os


class GenericHandler(object):
    '''
    Class for handling generic data files.
    '''
    def __init__(self, datasets):
        '''Constructor.'''
        self.dataset   = self.__class__.__name__.lower() # Name of class in lowercase format.
        self.datasets = datasets
        self.base_path = os.path.dirname(datasets[0])

    def get_files(self):
        '''Obtain all file paths for each of the corresponding sets (training and testing).'''
        file_list = []
        for idx, f in enumerate(self.datasets):
            new_file = {'image': f, 'mask': '', 'phase': 'Unk', 'info': {}, 'patient_id': idx}
            file_list.append(new_file)

        return file_list
