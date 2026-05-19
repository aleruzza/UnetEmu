import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import fforge
from fforge.inference.emulator import Emulator
from fforge.utils.utils import vr_norm, vaz_norm, generate_ict_128x128_disc_tri_slopes, nullv
import fforge.utils.units as u
from discminer.mining_utils import get_noise_mask, init_data_and_model
from discminer.model import keplerian
import h5py

#test setup
N = 20
Rp = 261 #au
phip = 57*np.pi/180
#testset_cmaps = np.load('cmaps_from_simulations_testset_parfile_R261_phi0.99_hd163296like3corr.npy')
savecmaps_filename = 'cmaps_emulations_small_alltraining.h5'

#setup emulator
labels = ["vphi", "vr"]
pths = [
    "../trained_models/vphi_256/model__epoch_1980_test_vaz_256.pth",
    "../trained_models/vr_256/model__epoch_1980_test_vr_256.pth",
]
params = [
    "../trained_models/vphi_256/params.py",
    "../trained_models/vr_256/params.py",
]

#functions used to denormalize the emulator output
norm_funcs = [vaz_norm, vr_norm]

emu = Emulator(
    model_pths=pths,
    labels=["vphi", "vr"],
    model_params=params,
    norm_funcs=norm_funcs,
    ict_gen=generate_ict_128x128_disc_tri_slopes,
)


def nullv(coord, Mstar=1.0, vel_sign=1, vsys=0, **kwargs):
    Mstar *= u.MSun
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    return vel_sign*np.sqrt(Mstar/R) * np.nan


#init_discminer
import get_hd163296_refmodel as getmo
model = getmo.get_model().model
model.velocity_func = emu.emulate_v3d

#Case specific parameters. Change to evaluate a different emulator or a different pipeline
testset_data = '../data/data/data/gas_tri_256/data.npy'
testset_params = '../data/data/data/gas_tri_256/run4.csv'
testset = np.load(testset_data)[:,[2,1],::-1]
testparams = pd.read_csv(testset_params, index_col=0)

xy = np.linspace(-3,3,256)
xx, yy = np.meshgrid(xy, xy)
rr = np.sqrt(xx**2+yy**2)
phi = np.arctan2(yy, xx)

mask1 = None
mask2 = None

from tqdm import tqdm
all_emulations = []
parameters = ['Alpha', 'PlanetMass', 'AspectRatio']
grids = {'Alpha': np.logspace(-4,-2,N), 'PlanetMass': np.logspace(-5,-2,N), 'AspectRatio': np.linspace(0.03, 0.1,N)}

#create file for saving
saving_file = h5py.File(savecmaps_filename, "a", libver='latest')
final_shape = (len(testparams), 22, 250, 250)

if 'emu_true_para' not in saving_file:
    saving_file.create_dataset(
        'emu_true_para',
        shape=final_shape,
        dtype="f4",
        chunks=(1, 22, 250, 250),
        compression=None
    )

saving_file.flush()
saving_file.swmr_mode = True


try:
    for j in range(len(testparams)):
        #setup emulator parameters and update emulator velocity function
        emu_params_sing = {
                "R_p": Rp * u.au,
                "phi_p": phip,
                "alpha": testparams['Alpha'].iloc[j],
                "h": testparams['AspectRatio'].iloc[j],
                "planetMass": testparams['PlanetMass'].iloc[j],
                "flaringIndex": testparams['FlaringIndex'].iloc[j],
                "extrap_vfunc": nullv,
                "mask_only_ppos": False
         }
            
        model.params['velocity'].update(emu_params_sing)  
        modelcube = model.make_model(make_convolve=False)
        saving_file['emu_true_para'][j] = modelcube.data
    
        if j%10==0:
            saving_file.flush()

finally:
    saving_file.close()
   

        
