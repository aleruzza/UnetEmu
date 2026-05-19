import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import fforge
from fforge.inference.emulator import Emulator
from fforge.utils.utils import vr_norm, vaz_norm, generate_ict_128x128_disc_tri_slopes, nullv
import fforge.utils.units as u


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

#Case specific parameters. Change to evaluate a different emulator or a different pipeline
testset_data = '../data/data/data/gas_tri_256/datatest.npy'
testset_params = '../data/data/data/gas_tri_256/testpara.csv'
testset = np.load(testset_data)[:,[2,1],::-1]
testparams = pd.read_csv(testset_params, index_col=0)

xy = np.linspace(-3,3,256)
xx, yy = np.meshgrid(xy, xy)
rr = np.sqrt(xx**2+yy**2)
phi = np.arctan2(yy, xx)

mask1 = (rr>0.5) & (rr<2)
mask2 = (rr>0.7) & (rr<1.3) & (phi>-0.2) & (phi<0.2)

from tqdm import tqdm
all_emulations = []
parameters = ['Alpha', 'PlanetMass', 'AspectRatio']
grids = {'Alpha': np.logspace(-4,-2,100), 'PlanetMass': np.logspace(-5,-2,100), 'AspectRatio': np.linspace(0.03, 0.1,100)}

mse_varp = {'Alpha': [], 'PlanetMass': [], 'AspectRatio': []}
for i in tqdm(range(len(testparams))):
    row = testparams.iloc[i]
    emu_params = {}
    for varied_par in parameters:
        for par in ['Alpha', 'PlanetMass', 'AspectRatio', 'SigmaSlope', 'FlaringIndex']:
            if par == varied_par:
                emu_params[par] = grids[par]
            else:
                emu_params[par] = np.ones(100)*row[par]
        all_emulations_vp = emu.emulate(alpha=emu_params['Alpha'],
                                        h=emu_params['AspectRatio'],
                                        planetMass=emu_params['PlanetMass'],
                                        sigmaSlope=emu_params['SigmaSlope'],
                                        flaringIndex=emu_params['FlaringIndex'], fields=['vr', 'vphi'])
        mse_varp[varied_par].append([ (((all_emulations_vp - testset[i].reshape(1,2,256,256))*mask)**2).mean(axis=(-1,-2)) for mask in [mask1,mask2]])

for par in parameters:
    mse_varp[par] = np.array(mse_varp[par])

np.save('mse_varp_testset.npy', mse_varp)


        