import numpy as np
import matplotlib.pyplot as plt
import emcee
import copy
from argparse import ArgumentParser

from fforge.inference.emulator import Emulator
from fforge.utils.utils import generate_ict_128x128_disc_tri_slopes, nullv
import fforge.utils.units as u
from fforge.utils.utils import hypot_func
from fforge.discminerIntegration import customDiscminerModel
from discminer.tools.utils import FrontendUtils

#this is needed to use torch in multithreading. Additionally, in discminer one should substitute the following lines:
#  'from multiprocessing import Pool' --> 'from multiprocessing.pool import ThreadPool as Pool'
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import torch


#useful to have these functions here to avoid problems with pickling
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
parser.add_argument('--tag-out', default='out', type=str, help='tag that will be used in the output filenames')
parser.add_argument('-p', '--parfile', default='parfile.json', type=str, help='discminer parfile of the disk to fit, the model defined by it will be fitted to the data changing only the velocity field')
parser.add_argument('-w', '--n-walkers', default=32, type=int, help='Number of walkers to use in the mcmc fit')
parser.add_argument('-s', '--n-steps', default=32, type=int, help='Number of steps to do in the mcmc fit')
parser.add_argument('-c', '--n-threads', default=None, type=int, help='Number of threads, default is the maximum available')
args = parser.parse_args()

#######################################################################################################################
# SET PARAMETERS TO FIT
#initial guesses
alpha = 1 #1e-3
h = 0.49  #0.05
mp = 0.57 #1e-3
sigma = -0.03 #1
fi = -0.03 #0.2
rp = 261*u.au #au
phip = 57*np.pi/180
log_emu_unc = -3.0
p0 = [h, mp, fi, log_emu_unc]

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
    },
    'likelihood':
    {
        'log_emu_unc': True
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
    },
    'likelihood':
    {
        'log_emu_unc': (-3.,0.)
    }
} 

#noise for fit likelihood
noise = 'auto'#6.2e-3
fit_normalized_values = True
extrapolation_vfunc = nullv
########################################################################################################################

#data
tag_out = args.tag_out #PREFERRED FORMAT: disc_mol_chan_program_extratags
parfile = args.parfile
tag_in = tag_out

###emulator
#setup and load the emulator. This is the last version.
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
nwalkers = args.n_walkers
nsteps = args.n_steps

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
model.params['velocity']['norm'] = not fit_normalized_values
model.params['velocity']['extrap_vfunc'] = extrapolation_vfunc
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
if noise=='auto':
    noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0) 

#set the custom likelihood to fit unc
from fforge.discminerIntegration import custom_ln_likelihood
model.ln_likelihood = custom_ln_likelihood.__get__(model)

model.__class__ = customDiscminerModel
model.__init_extra__()
model.populate(datacube.data, vchannels,
               p0_mean=p0)

#Run Emcee
model.run_mcmc(datacube.data, vchannels,
               p0_mean=p0, nwalkers=nwalkers, nsteps=nsteps,
               backend=backend,
               tag=tag_out,
               nthreads=args.n_threads, # If not specified considers maximum possible number of cores
               frac_stats=0.1,
               frac_stddev=0.1,
               noise_stddev=noise) 

print("Backend Final size: {0} steps".format(backend.iteration))

#***************************************
#SAVE SEEDING, BEST FIT PARS, AND ERRORS
vel_sign = model.params['velocity']['vel_sign']
model.mc_header.append('vel_sign')
np.savetxt('log_pars_%s_cube_%dwalkers_%dsteps.txt'%(tag_out, nwalkers, backend.iteration), 
           np.array([np.append(p0, vel_sign),
                     np.append(model.best_params, vel_sign),
                     np.append(model.best_params_errneg, 0.0),
                     np.append(model.best_params_errpos, 0.0)
           ]), 
           fmt='%.6f', header=str(model.mc_header))
