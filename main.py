import os
import wandb
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from params import params
from loader import TextImageDataset, PretrainDataset, scaleandlog
from create_model import create_nnmodel
    
    
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
                shuffle=True,
                rotaugm=params['rotaugm']
            )

    # data loader setup
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    #loading test set
    test_paradf = pd.read_csv(f'{params["datadir"]}/testpara.csv', index_col=0)
    slopes = np.array(test_paradf[['SigmaSlope']])
    #generating initial conditions
    x = np.linspace(-3, 3, 128)
    y = np.linspace(-3, 3, 128)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    ict = np.float32(r**(-slopes.reshape(-1,1,1))*((r<3) & (r>0.3)))
    #standardizing
    #means = ict.reshape(ict.shape[0], -1).mean(axis=1).reshape(-1,1,1)
    #stds = ict.reshape(ict.shape[0], -1).std(axis=1).reshape(-1,1,1)
    ict = np.expand_dims(scaleandlog(ict,1), axis=1)
    ict = torch.tensor(ict).to(device=params['device'])
    testparam = torch.tensor(np.float32(np.log10(np.array(test_paradf[['PlanetMass', 'AspectRatio', 'Alpha', 'InvStokes1', 'FlaringIndex']]))))
    testparam =  testparam.to(params['device'])
    xtest = torch.tensor(np.expand_dims(scaleandlog(np.load(f'{params["datadir"]}/datatest.npy'),1), axis=1)).to(params['device'])

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
    wandb.watch(model, criterion=loss_mse, log_freq=10)
    for ep in range(params['nepochs']):
        print(f'epoch {ep}')
        model.train() #setting model in training mode
        
        #linear lr decay
        if params['lr_decay']:
            optim.param_groups[0]['lr'] = params['lr']*(1-ep/params['nepochs'])
            
        pbar = tqdm(dataloader)
        mean_mse = np.array([])
        for i, (x, p, ic) in enumerate(pbar):
            model.train()
            optim.zero_grad() #reset the gradients
            x = x.to(params['device'])
            p = p.to(params['device'])
            ic = ic.to(params['device'])
            x_pred = model(ic, p)
            loss = loss_mse(x, x_pred).to(device=params['device'])
            loss.backward()
            mean_mse = np.append(mean_mse, [loss.item()])
            pbar.set_description(f'loss: {loss.item():.4f}')
            optim.step()
            
        if params['save_model']:
            if ep%params['savefreq']==0:
                torch.save(model.state_dict(), params['savedir'] + f"/model__epoch_{ep}_test_{params['name']}.pth")
        
        with torch.inference_mode():
            model.eval()
            x_pred_t = model(ict, testparam)
            xy = np.linspace(-3,3,128)
            mse_test = getmse(x_pred_t, xtest, xy, xy)
            wandb.log({'loss': mean_mse.mean(), 'epoch': ep, 'mse_test': mse_test})
        
def getmse(im1, im2, x, y):
    xx, yy = np.meshgrid(x,y)
    rr = np.sqrt(xx**2+yy**2)
    return (((im1-im2)**2)*((rr<3) & (rr>0.3))).mean()
    
    

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
    
    with wandb.init(project='emulator_unet', config=params, name=params['name']):
        #begin train
        if params['resume']:
            if not os.path.exists(params['resume_from']):
                print('Error! the model wich you want to resume from does not exist!\n Exiting...')
                exit()
            else:
                #TODO: implement possibility to resume
                exit()
        else:
            emulator = create_nnmodel(5, params['image_size']).to(device=params['device'])
            

        train(params=params, model=emulator)