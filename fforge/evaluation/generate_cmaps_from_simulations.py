from discminer.emulator import Emulator
from scipy.interpolate import griddata
import discminer.emulator as emulib
import torch
import numpy as np
import pandas as pd
from fforge.utils import units as u

# loading the datacube under exam
from discminer.core import Data
from sys import path as syspath
import json
from discminer import units as ucf
from tqdm import tqdm
import matplotlib.pyplot as plt
from discminer.mining_utils import get_noise_mask, init_data_and_model
from scipy.interpolate import RegularGridInterpolator
from discminer.diff_interp import get_griddata_sparse


def nullv(coord, Mstar=1.0, vel_sign=1, vsys=0, **kwargs):
    Mstar *= ucf.MSun
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    return vel_sign*np.sqrt(Mstar/R) * np.nan


#prepare the v2d func
def hypot_func(xx, yy): return np.sqrt(xx**2+yy**2)

    
def per_b(t):
    shape = t.shape
    t = t.flatten()
    t[t > np.pi] = t[t > np.pi] - 2 * np.pi
    t[t < -np.pi] = t[t < -np.pi] + 2 * np.pi
    return t.reshape(*shape)


def simulations_v2d(
        coord,
        R_p,
        phi_p,
        v3d, #shape (2, 256, 256)
        extrap_vfunc,
        mask_only_ppos=False,
        interp_3d='SPHERICAL',
        **extrap_kwargs,
    ):

        G = 6.67384e-11
        if "Mstar" in extrap_kwargs.keys():
            Mstar = extrap_kwargs["Mstar"]
        else:
            Mstar = 1

        if "vel_sign" in extrap_kwargs.keys():
            vel_sign = extrap_kwargs["vel_sign"]
        else:
            vel_sign = 1

        if vel_sign == -1:
            v3d = v3d[:,::-1].copy()
            v3d[0] = -1*v3d[0]

        x = np.linspace(-3, 3, 256)
        y = np.linspace(-3, 3, 256)
        xx, yy = np.meshgrid(x, y)
        rr = hypot_func(xx, yy)
        pp = np.arctan2(yy, xx)
        dom_mask = (rr > 0.4) & (rr < 3)
        rr_dom = rr[dom_mask]
        pp_dom = pp[dom_mask]
    
        rr_dom = rr_dom * R_p
        pp_dom = per_b(pp_dom + phi_p)
        v3d_dom = v3d[:,dom_mask]
        x_dom = rr_dom * np.cos(pp_dom)
        y_dom = rr_dom * np.sin(pp_dom)

        if "R" not in coord.keys():
            R = hypot_func(coord["x"], coord["y"])
        else:
            R = coord["R"]

        if "phi" not in coord.keys():
            phi = np.arctan2(coord["y"], coord["x"])
        else:
            phi = coord["phi"]

        if 'theta' not in coord.keys():
            theta = np.arccos(coord['z']/R)
        else:
            theta = coord['theta']

        if 'r' not in coord.keys():
            r = hypot_func(coord['z'], R)
        else:
            r = coord['r']
            
        if interp_3d == 'SPHERICAL':
            interpolator = get_griddata_sparse((x_dom, y_dom), (r*np.cos(phi), r*np.sin(phi)))

        vphi_interp = np.array(
            (
                interpolator(
                    v3d_dom[0].reshape(-1)
                )
                * np.sqrt(G * Mstar * u.MSun / R_p)
            )
            * 1e-3 #this is because we use km
       )
        
        vr_interp = np.array((
            interpolator(v3d_dom[1].reshape(-1))
            * 1e-3 * np.sqrt(G * Mstar * u.MSun / R_p)
        )
                             )

        mask = (R > 2.9 * R_p) | (R < 0.5 * R_p)
        mask_ppos = (R > 0.9* R_p) & (R < 1.1* R_p) & (phi<phi_p+0.5) & (phi>phi_p-0.5)
        
        vphi_interp[mask] = extrap_vfunc(coord, **extrap_kwargs)[mask]
        vr_interp[mask] = 0
        v3d_interp = np.concatenate(
            [
                np.expand_dims(vphi_interp, axis=0),
                np.expand_dims(vr_interp, axis=0),
                np.expand_dims(np.zeros(vphi_interp.shape), axis=0),
            ],
            axis=0,
        )

        if mask_only_ppos:
            v3d_interp[:,~mask_ppos] = np.nan

        #np.save('v3d.npy', v3d_interp)
        return v3d_interp


def generate_cmap(datav, model, R0, phi_p, convolve, **kwargs):

    #model.params["orientation"]["incl"] = np.pi / 4
    emu_params = {
            "R_p": R0 * ucf.au,
            "phi_p": phi_p,
            "v3d": datav,
            "extrap_vfunc": nullv
     }

    emu_params.update(kwargs)

    model.params['velocity'].update(emu_params)
    
    #emulate and generate cmap
    modelcube = model.make_model(make_convolve=convolve)

    #transform in an np array
    modelcube_arr = np.array(modelcube.data)
    return modelcube_arr


if __name__ == '__main__':
    
    Rp = 261 #au
    phi_p = 57*np.pi/180
    
    #load the testset
    testset_data = '../data/data/data/gas_tri_256/data.npy'
    testset = np.load(testset_data)[:,[1,2],::-1] #loads data of the testset with the correct sign

    #setup discminer
    import get_hd163296_refmodel as getmo
    model = getmo.get_model().model

    # setup discminer
    model.velocity_func = simulations_v2d

    cmaps = []
    for i in tqdm(range(testset.shape[0])):
        cmaps.append(generate_cmap(testset[i], model, Rp, phi_p, False, mask_only_ppos=False))

    cmaps = np.array(cmaps)

    np.save(f'cmaps_from_simulations_testset_parfile_R{Rp}_phi{phi_p:.2}_hd163296like3corr_train.npy', cmaps)