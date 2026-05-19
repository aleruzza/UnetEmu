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

#test setup
N = 20
Rp = 261 #au
phip = 57*np.pi/180
testset_cmaps = np.load('cmaps_from_simulations_testset_parfile_R261_phi0.99_hd163296like3corr.npy')

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
testset_data = '../data/data/data/gas_tri_256/datatest.npy'
testset_params = '../data/data/data/gas_tri_256/testpara.csv'
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

mse_varp = {'Alpha': [], 'PlanetMass': [], 'AspectRatio': []}

def get_maps(i, varied_par):
    row = testparams.iloc[i]
    emu_params = {}
    
    for par in ['Alpha', 'PlanetMass', 'AspectRatio', 'SigmaSlope', 'FlaringIndex']:
        if par == varied_par:
            emu_params[par] = grids[par]
        else:
            emu_params[par] = np.ones(N)*row[par]

    allparams_cmaps = []
    for j in range(N):
        #setup emulator parameters and update emulator velocity function
        emu_params_sing = {
                "R_p": Rp * u.au,
                "phi_p": phip,
                "alpha": emu_params['Alpha'][j],
                "h": emu_params['AspectRatio'][j],
                "planetMass": emu_params['PlanetMass'][j],
                "flaringIndex": emu_params['FlaringIndex'][j],
                "extrap_vfunc": nullv,
                "mask_only_ppos": False
         }
    
        model.params['velocity'].update(emu_params_sing)  
        modelcube = model.make_model(make_convolve=False)

        allparams_cmaps.append(modelcube.data)
    allparams_cmaps = np.array(allparams_cmaps)

    return allparams_cmaps
