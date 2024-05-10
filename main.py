import os
import copy
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from params import params
from loader import TextImageDataset, PretrainDataset
from create_model import create_nnmodel
from torch.utils.tensorboard import SummaryWriter
    
    
def train(params, model):
    
    # initialize the dataset
    #if pretrain load the pretraining dataset
    if params['pretrain']:
        dataset = PretrainDataset(
            folder=params['datadir'],
            image_size=128,
            shuffle=True,
            n_param=params['n_param'],
            n_pretrain = params['n_pretrain']
        )
        test_param = None
    else:
        #targets
        dataset = TextImageDataset(
                folder=params['datadir'],
                image_size=params['image_size'],
                uncond_p=params['drop_prob'], # only used when drop_para=True
                shuffle=True,
                n_param=params['n_param'],
                drop_para=True if params['cond']==True else False
            )

    # data loader setup
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    #training loop
    params_to_optimize = [
            {'params': model.parameters()}
        ]
    
    # number of parameters to be trained
    number_of_params = sum(x.numel() for x in model.parameters())
    print(f"Number of parameters for unet: {number_of_params}")
    
    #define the loss function
    loss_mse = nn.MSELoss()
    
    length = len(dataloader)
    
    #initialize optimizer
    optim = torch.optim.Adam(params_to_optimize, lr=params['lr'])

    #loop
    for ep in range(params['nepochs']):
        print(f'epoch {ep}')
        model.train() #setting model in training mode
        
        #linear lr decay
        if params['lr_decay']:
            optim.param_groups[0]['lr'] = params['lr']*(1-ep/params['nepochs'])
            
        pbar = tqdm(dataloader)
        for i, (x, p, ic) in enumerate(pbar):
            optim.zero_grad() #reset the gradients
            x = x.to(params['device'])
            x_pred = model(ic, p)
            loss = loss_mse(x, x_pred)
            loss.backward()
            
            pbar.set_description(f'loss: {loss.item():.4f}')
            optim.step()
            
            if params['save_model']:
                if ep%params['savefreq']==0:
                    torch.save(model.state_dict(), params['savedir'] + f"/model__epoch_{ep}_test_{params['name']}.pth")
                    

def test(emulator):
    """TODO: implement this function

    Parameters
    ----------
    emulator : _type_
        _description_
    """
    #parameters for test set
    test_paradf = pd.read_csv(f'data/testpara.csv', index_col=0)
    test_param = torch.tensor(np.float32(np.log10(np.array(test_paradf[['PlanetMass', 'AspectRatio', 'Alpha', 'InvStokes1', 'FlaringIndex']]))))
    test_param =  test_param.to(params['device'])
    
    #initial conditions
    slopes = np.array(test_paradf['SigmaSlope'])
    x = np.linspace(-4, 4, 128)
    y = np.linspace(-4, 4, 128)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    ic_input_tests = torch.tensor(np.float32(r**(-slopes.reshape(-1,1,1))*((r<4) & (r>0.4)).astype(float)))
    means = ic_input_tests.reshape(ic_input_tests.shape[0], -1).mean(axis=1)
    stds = ic_input_tests.reshape(ic_input_tests.shape[0], -1).std(axis=1)
    ic_input_tests = (ic_input_tests-means)/stds
    
if __name__ == "__main__":
    
    #checking if exists and creating output directory if it does not
    if os.path.exists(params['savedir']):
        if params['Override']:
            print('Saving directory exists, overriding old data as instructed.')
        else:
            print('WARNING! -> saving directory already exists, please run with Override=True')
            exit()
    else:
        os.mkdir(params['savedir'])

    #checking file with parameter history and adding this run
    if os.path.exists('parahist.csv'):
        oldpara = pd.read_csv('parahist.csv', index_col=0)
        params['index'] = oldpara.index[-1]+1
        newparafile = pd.concat([oldpara, pd.DataFrame([params]).set_index('index')])
    else:
        params['index'] = 0
        newparafile = pd.DataFrame([params]).set_index('index')
    newparafile.to_csv('parahist.csv')
    
    #begin train
    if params['resume']:
        if not os.path.exists(params['resume_from']):
            print('Error! the model wich you want to resume from does not exist!\n Exiting...')
            exit()
        else:
            #TODO: implement possibility to resume
            exit()
    else:
        emulator = create_nnmodel(5, params['image_size'])
        

    train(params=params, model=emulator)