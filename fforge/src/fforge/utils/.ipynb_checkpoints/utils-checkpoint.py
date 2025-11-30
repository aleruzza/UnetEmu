import numpy as np
import sys
import importlib.util
from pathlib import Path
from . import units as u
import torch

def nullv(coord, Mstar=1.0, vel_sign=1, vsys=0):
    Mstar *= u.MSun
    if 'R' not in coord.keys(): R = hypot_func(coord['x'], coord['y'])
    else: R = coord['R']
    return vel_sign*np.sqrt(Mstar/R) * np.nan


def hypot_func(x, y):
    return np.sqrt(x**2 + y**2)


def load_params(params_path):
    """Dynamically loads the params.py file and extracts the 'params' dictionary."""
    params_path = Path(params_path).resolve()
    module_name = "params_module"

    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")

    spec = importlib.util.spec_from_file_location(module_name, params_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "params"):
        raise AttributeError("params.py does not contain a 'params' dictionary")

    return module.params


def generate_ict_128x128_disc_tri(slopes, dimension):
    x = np.linspace(-3, 3, dimension)
    y = np.linspace(-3, 3, dimension)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    vaz_ict = np.float32(r ** (-0.5) * ((r < 3) & (r > 0.4)))
    vaz_ict = np.expand_dims(
        np.repeat(np.expand_dims(vaz_ict, 0), len(slopes), axis=0), 1
    )
    vr_ict = np.zeros(vaz_ict.shape)
    dens_ict = generate_ict_128x128_disc(slopes, dimension=dimension, nonorm=True)
    ict = np.concatenate([dens_ict, vaz_ict, vr_ict], axis=1)
    return np.float32(ict)


def generate_ict_128x128_disc_tri_slopes(slopes, dimension):
    dens_ict = generate_ict_128x128_disc(slopes, dimension=dimension, nonorm=True)
    ict = np.concatenate([dens_ict, dens_ict.copy(), dens_ict.copy()], axis=1)
    return np.float32(ict)


def generate_ict_128x128_disc(slopes, dimension, nonorm=False):
    # generating initial conditions
    x = np.linspace(-3, 3, dimension)
    y = np.linspace(-3, 3, dimension)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    ict = np.float32(r ** (-slopes.reshape(-1, 1, 1)) * ((r < 3) & (r > 0.4)))
    if not nonorm:
        ict = np.float32(ict)
    ict = np.expand_dims(ict, axis=1)
    return ict


def norm_labels(labels):
    # ['PlanetMass', 'AspectRatio', 'Alpha',  'FlaringIndex']
    max = np.array([1e-2, 0.1, 0.01, 0.35])
    min = np.array([1e-5, 0.03, 1e-4, 0])
    for i in [0, 2]:
        labels[:, i] = np.log10(labels[:, i])
        max[i] = np.log10(max[i])
        min[i] = np.log10(min[i])
    labels = 2 * (labels - min) / (max - min) - 1
    return labels


def vr_norm(data):
    return data * 1e-2


def vaz_norm(data):
    xy = np.linspace(-3, 3, data.shape[-1])
    xx, yy = np.meshgrid(xy, xy)
    rr = hypot_func(xx, yy)
    return data * 1e-2 + torch.Tensor(rr)**-0.5
