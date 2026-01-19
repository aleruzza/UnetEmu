from discminer.emulator import Emulator
import discminer.emulator as emulib
import torch
import numpy as np
import pandas as pd

# loading the datacube under exam
from discminer.core import Data
from sys import path as syspath
from astropy import units as u
import json
from discminer import units as ucf
from tqdm import tqdm
import matplotlib.pyplot as plt
from discminer.mining_utils import get_noise_mask, init_data_and_model

import torch


def vr_norm(data):
    return data * 1e-2


xy = np.linspace(-3, 3, 256)
xx, yy = np.meshgrid(xy, xy)
rr = np.sqrt(xx**2 + yy**2)
rr = torch.tensor(rr, dtype=torch.float)


def vaz_norm(data):
    return data * 1e-2 + rr**-0.5


#setup emulator
labels = ["vphi", "vr"]
pths = [
    "/home/aleruzza/UNI/SCIENCE/UnetEmu/outputs/vphi_256/model__epoch_1980_test_vaz_256.pth",
    "/home/aleruzza/UNI/SCIENCE/UnetEmu/outputs/vr_256/model__epoch_1980_test_vr_256.pth",
]
params = [
    "../outputs/vphi_256/params.py",
    "../outputs/vr_256/params.py",
]
norm_funcs = [vaz_norm, vr_norm]
emu = Emulator(
    model_pths=pths,
    labels=["vphi", "vr"],
    model_params=params,
    norm_funcs=norm_funcs,
    ict_gen=emulib.generate_ict_128x128_disc_tri_slopes,
)

# setup discminer
datacube, model = init_data_and_model(
    Rmin=0,
    Rmax=1,
    parfile="parfile.json",
)
model.velocity_func = emu.emulate_v2d
model.params["orientation"]["incl"] = np.pi / 4



#open table of parameters for which we have to generate a cmap
params_test = pd.read_csv("../data/data/gas_tri_256/testpara.csv")

#emulate and generate cmap
modelcubes = []
for simid in tqdm(range(0, 233)):
    emu_params = {
        "alpha": params_test.loc[simid, "Alpha"],
        "h": params_test.loc[simid, "AspectRatio"],
        "planetMass": params_test.loc[simid, "PlanetMass"],
        "sigmaSlope": params_test.loc[simid, "SigmaSlope"],
        "flaringIndex": params_test.loc[simid, "FlaringIndex"],
        "R_p": 400 * ucf.au,
        "phi_p": np.pi / 4,
        "extrap_vfunc": model.keplerian,
    }
    model.params["velocity"].update(emu_params)
    modelcube = model.make_model(make_convolve=False)
    modelcube = model.make_model(make_convolve=False)
    modelcube.convert_to_tb()
    modelcubes.append(modelcube.damta)

#transform in a np array
modelcubes_arr = np.array(modelcubes)
np.save("modelcubes_emu.npy", modelcubes_arr)
