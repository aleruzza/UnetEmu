import time
from pathlib import Path
from random import randint, random
import torch as th
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

################### Normalization functions ###################################
def scaleandlog(data, scale):
    data = np.nan_to_num(data)
    return np.log10(1 + data/scale)


############ functions that read numpy array data ##############################

def get_image_files_narray(base_path):
    image_files = np.load(f'{base_path}/data.npy')
    return image_files

def getlabels(dataframe):
    dataframe[['PlanetMass', 'Alpha', 'InvStokes1']] = np.log10(dataframe[['PlanetMass', 'Alpha', 'InvStokes1']])
    labels = np.array(dataframe[['PlanetMass', 'AspectRatio', 'Alpha', 'InvStokes1', 'FlaringIndex']])
    return labels

def get_labels_narray(base_path):
    paradf = pd.read_csv(f'{base_path}/run4.csv', index_col=0)
    labels = getlabels(paradf)
    
    #initial conditions
    slopes = np.array(paradf['SigmaSlope'])
    x = np.linspace(-3, 3, 128)
    y = np.linspace(-3, 3, 128)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    
    #generating initial conditions
    ic_inputs = r**(-slopes.reshape(-1,1,1))*((r<3) & (r>0.3)).astype(float)
    ic_inputs = scaleandlog(ic_inputs, 1)
    #standardizing
    #means = ic_inputs.reshape(ic_inputs.shape[0], -1).mean(axis=1).reshape(-1,1,1)
    #stds = ic_inputs.reshape(ic_inputs.shape[0], -1).std(axis=1).reshape(-1,1,1)
    #ic_inputs = (ic_inputs-means)/stds
    
    return labels, ic_inputs


def get_pretraining_data(base_path, n=10):
    dataset = np.load(f'{base_path}/swe_data.npy')
    dataset = dataset.reshape(-1, 128, 128)
    np.random.shuffle(dataset)
    return dataset[0:n]

#################################################################################

###################### Datasets #################################################

class PretrainDataset(Dataset):
    """Dataset for pretraining

 
    """
 
    def __init__(
        self,
        folder="",
        image_size=128,
        shuffle=False,
        n_param =5,
        n_pretrain=100
    ):
        """Init

        Parameters
        ----------
        folder : str, optional
            folder where the .npy files are stored, by default ""
        image_size : int, optional
            pixel dimension of the images in the dataset, by default 128
        shuffle : bool, optional
            enables shuffling of the dataset, by default False
        n_param : int, optional
            number of conditional parameters for which the model is built.
            Note that during pretraining all these parameters
            will be set to 0, by default 6
        """        
        super().__init__()
        folder = Path(folder)
        self.data = get_pretraining_data(folder, n=n_pretrain)
        self.shuffle = shuffle
        self.prefix = folder
        self.image_size = image_size
        self.n_param = n_param

    def __len__(self):
        return len(self.data)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        tokens = np.float32(np.array([0]).repeat(self.n_param)) #array([0,0]) for uncondional training
        original_image = np.float32(self.data[ind])
        arr = np.expand_dims(original_image,axis=0) # only one channel
        return th.tensor(arr),th.tensor(np.float32(tokens))

    

class TextImageDataset(Dataset):
    def __init__(
        self,
        folder="",
        image_size=64,
        shuffle=False,
        rotaugm=False
    ):
        super().__init__()
        folder = Path(folder)
        self.data = get_image_files_narray(folder)
        self.labels, self.ics = get_labels_narray(folder)
        self.rotaugm = rotaugm
        self.shuffle = shuffle
        self.prefix = folder
        self.image_size = image_size
        
        if rotaugm:
            self.transform = T.RandomRotation(90)

    def __len__(self):
        return len(self.labels)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)


    def __getitem__(self, ind):
        original_image = np.float32(self.data[ind])
        arr = scaleandlog(np.expand_dims(original_image,axis=0), 1e-5)
        ic = np.expand_dims(np.float32(self.ics[ind]), axis=0)
        arr = th.tensor(arr)
        if self.rotaugm:
            arr = self.transform(arr)
        return arr, th.tensor(np.float32(self.labels[ind])), th.tensor(ic)
    

