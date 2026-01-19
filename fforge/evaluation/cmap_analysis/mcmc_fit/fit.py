from discminer.core import Data
from discminer.disc2d import Model
from discminer.emulator import Emulator
import discminer.emulator as emulib
import torch
from discminer.mining_utils import init_data_and_model
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
import emcee

from argparse import ArgumentParser

def hypot_func(x,y):
    return np.sqrt(x**2+y**2)

parser = ArgumentParser(prog='Handle emcee backend', description='Handle emcee backend')
parser.add_argument('-b', '--backend', default=1, type=int, choices=[0, 1], help="If 0, create new backend. If 1, reuse existing backend")
args = parser.parse_args()

#*********************
#REQUIREd DEFINITIONS
#*********************
file_data = '' #fits file to fit
tag_out = '' #PREFERRED FORMAT: disc_mol_chan_program_extratags
tag_in = tag_out

nwalkers = 150
nsteps = 15000

dpc = 162.0*u.pc
vel_sign = -1 #Rotation direction: -1 or 1

Rmax = 1000*u.au #Model maximum radius

#*********
#READ DATA
#*********
datacube = Data(file_data, dpc) #Read data and convert to Cube object
vchannels = datacube.vchannels

au_to_m = u.au.to('m')

#****************************
#INIT MODEL AND PRESCRIPTIONS
#****************************
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
_, model = init_data_and_model(
    Rmin=0,
    Rmax=1,
    parfile="parfile.json",
)
model.velocity_func = emu.emulate_v2d
model.params["orientation"]["incl"] = np.pi / 4            

def intensity_powerlaw_rout(coord, I0=30.0, R0=100*au_to_m, p=-0.4, z0=100*au_to_m, q=0.3, Rout=200):
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    z = coord['z']
    A = I0*R0**-p*z0**-q
    Ieff = np.where(R<=Rout*au_to_m, A*R**p*np.abs(z)**q, 0.0)
    return Ieff

def z_upper(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

def z_lower(coord, z0, p, Rb, q, R0=100):
    R = coord['R']/au_to_m
    return -au_to_m*(z0*(R/R0)**p*np.exp(-(R/Rb)**q))

model.z_upper_func = z_upper
model.z_lower_func = z_lower
model.line_profile = model.line_profile_bell
model.intensity_func = intensity_powerlaw_rout
model.params['velocity']['extrap_vfunc'] = model.keplerian

#If not redefined, intensity and linewidth are powerlaws 
 #of R and z by default, whereas lineslope is constant.
  #See Table 1 of discminer paper 1 (izquierdo+2021).

#****************************
#SET FREE PARS AND BOUNDARIES
#****************************
# If True, parameter is allowed to vary freely.
#  If float, parameter is fixed to the value provided.

model.mc_params['velocity']['vel_sign'] = vel_sign 
model.mc_params['velocity']['alpha']= True
model.mc_params['velocity']['h'] = True
model.mc_params['velocity']['planetMass'] = True
model.mc_params['velocity']['sigmaSlope'] = True
model.mc_params['velocity']['flaringIndex'] = True
model.mc_params['velocity']['R_p'] = True
model.mc_params['velocity']['phi_p'] = True
                                   
# Boundaries of user-defined attributes must be defined here.
# Boundaries of attributes existing in discminer can be modified here, otherwise default values are taken.

model.mc_boundaries['velocity']['alpha'] = (1e-4,1e-2)
model.mc_boundaries['velocity']['h'] = (0.03, 0.1)
model.mc_boundaries['velocity']['planetMass'] = (1e-5, 1e-2)
model.mc_boundaries['velocity']['sigmaSlope'] = (0.5, 1.2)
model.mc_boundaries['velocity']['flaringIndex'] = (0, 0.35)
model.mc_boundaries['velocity']['R_p'] = (50, 300)*au_to_m
model.mc_boundaries['velocity']['phi_p'] = (-np.pi, np.pi)

#***************************
#INITIAL GUESS OF PARAMATERS
#***************************
alpha = 1e-3
h = 0.05
mp = 1e-3
sigma = 1
fi = 0.2
rp = 250
phip = 0

p0 = [alpha,
      h,
      mp,
      sigma,
      fi,
      rp,
      phip
]

#********
#RUN MCMC
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

print("Backend Initial size: {0} steps".format(backend.iteration))

# Noise in each pixel is stddev of intensity from first and last 5 channels 
noise = np.std( np.append(datacube.data[:5,:,:], datacube.data[-5:,:,:], axis=0), axis=0) 

#Run Emcee
model.run_mcmc(datacube.data, vchannels,
               p0_mean=p0, nwalkers=nwalkers, nsteps=nsteps,
               backend=backend,
               tag=tag_out,
               #nthreads=96, # If not specified considers maximum possible number of cores
               frac_stats=0.1,
               frac_stddev=1e-2,
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
