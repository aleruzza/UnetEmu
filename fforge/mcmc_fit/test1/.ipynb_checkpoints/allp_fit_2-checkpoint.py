import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
import numpy as np
import matplotlib.pyplot as plt
import emcee
import copy
from argparse import ArgumentParser

from fforge.inference.emulator import Emulator
from fforge.utils.utils import generate_ict_128x128_disc_tri_slopes, nullv
import fforge.utils.units as u

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

#import multiprocessing
#multiprocessing.set_start_method('spawn', force=True)

def vr_norm(data):
    return data * 1e-2


def vaz_norm(data):
    xy = np.linspace(-3, 3, data.shape[-1])
    xx, yy = np.meshgrid(xy, xy)
    rr = hypot_func(xx, yy)
    return data * 1e-2 + torch.Tensor(rr)**-0.5


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v


parser = ArgumentParser(prog='Handle emcee backend', description='Handle emcee backend')
parser.add_argument('-b', '--backend', default=0, type=int, choices=[0, 1], help="If 0, create new backend. If 1, reuse existing backend")
args = parser.parse_args()


#*********************
#REQUIRED DEFINITIONS
#*********************

#data
tag_out = 'test1' #PREFERRED FORMAT: disc_mol_chan_program_extratags
parfile = 'parfile.json'
tag_in = tag_out

#emulator settings
#setup emulator
labels = ["vphi", "vr"]
pths = [
    "../../trained_models/vphi_256/model__epoch_1980_test_vaz_256.pth",
    "../../trained_models/vr_256/model__epoch_1980_test_vr_256.pth",
]
params = [
    "../../trained_models/vphi_256/params.py",
    "../../trained_models/vr_256/params.py",
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

#mcmc fit settings
nwalkers = 6
nsteps = 15000

#set parameters to fit
#initial guesses
alpha = 0 #1e-3
h = 0.49  #0.05
mp = 0.57 #1e-3
sigma = -0.03 #1
fi = -0.03 #0.2
rp = 261*u.au #au
phip = 57*np.pi/180
p0 = [h, mp, fi]

mc_params = {
    'velocity': 
    {
        'alpha': alpha,
        'h': True,
        'planetMass': True,
        'sigmaSlope': None,
        'flaringIndex': True,
        'R_p': rp,
        'phi_p': phip
    }
}

mc_boundaries = {
    'velocity': 
    {
        'alpha': alpha,#(-1,1),
        'h': (-1,1),
        'planetMass': (-1,1),
        #'sigmaSlope': (0.5,1.2),
        'flaringIndex': (-1, 1),
        'R_p': rp,#(300*u.au, 500*u.au),
        'phi_p': phip, #(-np.pi, np.pi)
    }
} 

#noise for fit likelihood
noise = 6.2e-3

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************

#setup emulator

# setup discminer
#init_discminer
from discminer.mining_utils import init_data_and_model
datacube, model = init_data_and_model(
    Rmin=1.0,
    parfile='parfile.par',
)
model.prototype = False

model.velocity_func = emu.emulate_v3d

vchannels = model.vchannels

model.params['velocity']['norm'] = False
model.params['velocity']['extrap_vfunc'] = nullv

model.mc_params = copy.deepcopy(model.params)
deep_update(model.mc_params, mc_params)
model.mc_boundaries.update(mc_boundaries)

#********
#RUN MCMC
#********
#********
# Set up the emcee backend
filename = "backend_%s.h5"%tag_out
backend = None

#try and except statement failing with FileNotFoundError/OSError
if args.backend:
    #Succesive runs
    backend = emcee.backends.HDFBackend(filename)
else:
    #First run: Initialise backend
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, len(p0))
    
# Noise in each pixel is stddev of intensity from first and last 5 channels 
#noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0) 

#Run Emcee
import emcee

model.run_mcmc(datacube.data, vchannels,
               p0_mean=p0, nwalkers=nwalkers, nsteps=nsteps,
               backend=backend,
               tag=tag_out,
               nthreads=1, # If not specified considers maximum possible number of cores
               frac_stats=0.1,
               frac_stddev=0.1,
               noise_stddev=noise) 

print("Backend Final size: {0} steps".format(backend.iteration))

#***************************************
#SAVE SEEDING, BEST FIT PARS, AND ERRORS
model.mc_header.append('vel_sign')
np.savetxt('log_pars_%s_cube_%dwalkers_%dsteps.txt'%(tag_out, nwalkers, backend.iteration), 
           np.array([np.append(p0, vel_sign),
                     np.append(model.best_params, vel_sign),
                     np.append(model.best_params_errneg, 0.0),
                     np.append(model.best_params_errpos, 0.0)
           ]), 
           fmt='%.6f', header=str(model.mc_header))
