import numpy as np
import losses
name = 'mmodes'
################### Normalization functions ###################################
def scaleandlog(data, scale):
    data = np.nan_to_num(data)
    return np.log10(1 + data/scale)

def nonorm(data, scale):
    return data/scale

######################################################################
params = {
    'name': name,  
    'device': 'cuda', 
    'nepochs': 800,
    'lr': 1e-4,
    'save_model': True,
    'savedir': f'./outputs/{name}',
    'datadir': f'./mmodes_data/',
    'mode': 'mdeco',
    'Override': True,
    'savefreq': 20,
    'cond': True,
    'lr_decay': False,
    'resume': False,
    'sample_freq': 10, 
    'batch_size': 32,
    'rotaugm': False,
    'image_size': 128,
    'logima_freq': 20,
    'loss': losses.MSEandFFT(),
    'norm': nonorm,
    'n_test_log_images': 10,
    'num_channels': 96,
    'channel_mult': "1, 1, 2, 3, 4",
    'num_res_blocks': 3,
    'pretrain': False,
    'n_param' : 5,
    'n_pretrain': 10000 #note: it must be <101,000
}

