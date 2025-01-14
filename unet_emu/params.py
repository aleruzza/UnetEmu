import numpy as np
import losses
import torch
name = 'tri'
################### Normalization functions ###################################
def scaleandlog(data, scale):
    data = np.nan_to_num(data)
    return np.log10(1 + data/scale)

def nonorm(data, scale):
    return data/scale

def norm_labels(labels):
    #['PlanetMass', 'AspectRatio', 'Alpha', 'InvStokes1', 'FlaringIndex']
    max = np.array([1e-2, 0.1, 0.01, 1e3, 0.35])
    min = np.array([1e-5, 0.03, 1e-4, 10, 0])
    for i in [0, 2, 3]:
        labels[:, i] = np.log10(labels[:,i])
        max[i] = np.log10(max[i])
        min[i] = np.log10(min[i])
    labels = 2*(labels-min)/(max-min) - 1
    return labels

def norm_labels_gas(labels):
    #['PlanetMass', 'AspectRatio', 'Alpha',  'FlaringIndex']
    max = np.array([1e-2, 0.1, 0.01,  0.35])
    min = np.array([1e-5, 0.03, 1e-4,0])
    for i in [0, 2]:
        labels[:, i] = np.log10(labels[:,i])
        max[i] = np.log10(max[i])
        min[i] = np.log10(min[i])
    labels = 2*(labels-min)/(max-min) - 1
    return labels

def norm_cube_log(data, scale):
    shape = [1,3,1,1] if len(data.shape)==4 else [3,1,1]
    scale = np.array([scale, 1, 1]).reshape(shape)
    data = np.nan_to_num(data)
    return data/scale


######################################################################
params = {
    'name': name,  
    'device': 'cuda', 
    'nepochs': 801,
    'lr': 1e-4,
    'save_model': True,
    'savedir': f'../outputs/{name}',
    'datadir': f'../data/gas_tri/',
    'mode': '128x128_disc_tri',
    'Override': True,
    'savefreq': 20,
    'cond': True,
    'lr_decay': False,
    'resume': False,
    'periodic_bound_x': False,
    'sample_freq': 10, 
    'batch_size': 32,
    'rotaugm': False,
    'image_size': 128,
    'logima_freq': 20,
    'loss': torch.nn.MSELoss(),
    'unc': False,
    'norm': norm_cube_log,
    'scale': 1e-3,
    'norm_labels': norm_labels_gas,
    'n_test_log_images': 50,
    'num_channels': 96,
    'channel_mult': "1, 1, 2, 3, 4",
    'num_res_blocks': 3,
    'pretrain': False,
    'n_param' : 4,
    'infer_labels': ['PlanetMass', 'AspectRatio', 'Alpha', 'FlaringIndex'],
    'n_pretrain': 10000 #note: it must be <101,000
}

