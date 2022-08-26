import numpy as np
from torch.utils.data import Dataset

class Radardata(Dataset):
    
    def __init__(self, n_points,isValidation):

     
        self.inputs = np.load('input_files/inputs.npy')
        self.labels = np.load('input_files/labels.npy')
        self.val_indices = np.load('input_files/val_indices.npy')
        self.train_indices = np.load('input_files/train_indices.npy')
        
        self.isValidation = isValidation

    def __len__(self):
        if(self.isValidation):
            return len(self.val_indices)
        else:
            return len(self.train_indices)

    def __getitem__(self, idx):

        if(self.isValidation):
            ID = self.val_indices[idx]
        else:
            ID = self.train_indices[idx]
        points = self.inputs[ID,:,:]
        labels = self.labels[ID]

        sample = {'points': points, 'labels': labels}
        return sample
