from discminer.model import ReferenceModel
from astropy import units as u
from discminer.cart import *
from discminer.mining_utils import get_noise_mask, init_data_and_model

from radio_beam import Beam
#see Pinte+2018 and Izquierdo+2022

npix=250

init_params = { #these are from Izquierdo+2022
            'velocity': {
                'Mstar': 1.97,
                'vel_sign': +1,
                'vsys': 5.77
            },
            'orientation': {
                'incl': 45.71*np.pi/180,
                'PA': 42.35*np.pi/180,
                'xc': 0.0,
                'yc': 0.0
            },
            'intensity': {
                'I0': 0.3034,
                'p': -4.16, 
                'q': 3.68,
                'Rout': 380
            },
            'linewidth': {
                'L0': 0.08, 
                'p': 0.86, 
                'q': -1.38
            }, 
            'lineslope': {
                'Ls': 1.85, 
                'p': 0.21, 
                'q': 0.0
            },
            'height_upper': {
                'z0': 29.78,
                'p': 1.21,
                'Rb': 4.36,
                'q': 1.98
            },
            'height_lower': {
                'z0': 19.91,
                'p': 1.09,
                'Rb': 0.03,
                'q': 4.18
            }
        }

init_funcs = {
            'velocity': keplerian_vertical,
            'z_upper' : z_upper_powerlaw,
            'z_lower' : z_lower_powerlaw,
            'intensity' : intensity_powerlaw_rout,
            'linewidth' : linewidth_powerlaw,
            'lineslope' : lineslope_powerlaw,
            'line_profile' : line_profile_bell,
            'line_uplow' : line_uplow_mask,                        
        }

model = ReferenceModel(
    disc = 'hd163296_prot',
    Rmax = 380*u.au,
    dpc = 101.5*u.pc,
    npix = npix,
    vchannels = np.linspace(2.44, 9.16, 22), #km/s, only the 22 central channels
    beam = Beam(
                major=0.104*u.arcsec,
                minor=0.094*u.arcsec,
                pa=-80.2*u.deg
            ),
    init_params = init_params, 
    init_funcs = init_funcs
)

def get_model():
    return model