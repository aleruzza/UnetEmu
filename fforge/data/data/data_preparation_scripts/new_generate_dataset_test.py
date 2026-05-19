import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os

params = pd.read_csv('indacorun/paratest.csv', index_col=0)
pcoord = np.load('pcoord_test.npy')

def per_b(t):
    shape = t.shape
    t = t.flatten()
    t[t>np.pi] = t[t>np.pi]-2*np.pi
    t[t<-np.pi] = t[t<-np.pi]+2*np.pi
    return t.reshape(*shape)

x = np.linspace(-3, 3, 256)
y = np.linspace(-3, 3, 256)

xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx**2+yy**2)
theta = np.arctan2(yy, xx)
#rr = np.logspace(np.log10(0.4), np.log10(3), 128)
#tt = np.linspace(-np.pi, np.pi, 512)

#r, theta = np.meshgrid(rr, tt)
ids = []
dataset = []
j=0
for i, row in tqdm(params.iterrows()):
    theta_p = np.arctan2(pcoord[j, 1], pcoord[j, 0])
    single_data = []
    for filename in ['gasdens30.dat', 'gasvx30.dat', 'gasvy30.dat']:
        r_g = np.logspace(np.log10(0.4), np.log10(row['rout']), row['ny'].astype('int'))
        theta_g = np.linspace(-np.pi, np.pi, row['nx'].astype('int'))
        rho_d = np.fromfile(f'indacorun/outputs/out_{i:05}/{filename}').reshape(row['ny'].astype('int'), row['nx'].astype('int'))
        rho = RegularGridInterpolator((r_g, theta_g), rho_d, bounds_error=False, method='linear')
        data = np.nan_to_num(rho((r, per_b(theta+theta_p))))
        single_data.append(data.reshape(*data.shape,1))

    #data = np.log10(1+data)
    #data = (data-data.mean())/data.std()
    dataset.append(np.concatenate(single_data, axis=-1))
    ids.append(i)
    j+=1
dataset = np.array(dataset)

np.save('gas_tri_ids_256_test.npy', np.array(ids))
np.save('gas_tri_256_test.npy', np.array(dataset))
