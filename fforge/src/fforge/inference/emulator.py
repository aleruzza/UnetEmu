import numpy as np

from ..models.create_model import create_nnmodel
import torch
from scipy.interpolate import griddata
from ..utils import units as u
from ..utils.utils import hypot_func, load_params, norm_labels, generate_ict_128x128_disc
from discminer.diff_interp import get_griddata_sparse


class BaseEmulator:

    def __init__(self, model_pth="", model_para={}, device="cpu", norm_func=None):
        self.params = model_para
        self.device = device
        self.emulator = create_nnmodel(
            n_param=self.params["n_param"],
            image_size=self.params["image_size"],
            num_channels=self.params["num_channels"],
            num_res_blocks=self.params["num_res_blocks"],
            channel_mult=self.params["channel_mult"],
            mode=self.params["mode"],
            unc=self.params["unc"],
        ).to(device=torch.device(self.device))
        dataem = torch.load(model_pth, map_location=torch.device(self.device))
        self.emulator.load_state_dict(dataem)
        self.norm_func = norm_func if norm_func is not None else lambda value: value

    def emulate(self, ic, labels):
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
        ic = torch.tensor(ic, dtype=torch.float32, device=self.device)
        emulation = self.emulator(ic, labels)
        return self.norm_func(emulation)
    
    def __call__(self, ic, labels):
        return self.emulate(ic, labels)


class Emulator:

    def __init__(
        self,
        model_pths=[],
        model_params=[],
        labels=["dens", "vphi", "vr", "vz"],
        device="cpu",
        ict_gen=generate_ict_128x128_disc,
        ict_comp_dict = {'dens':0, 'vphi':0 , 'vr': 0},
        norm_funcs = [None, None, None, None]
    ):
        self.device = device
        self.emulators = {}
        self.ict_gen = ict_gen
        self.max_image_size = 0
        self.ict_comp_dict = ict_comp_dict
        for i, key in enumerate(labels):
            params = load_params(model_params[i])
            if params['image_size'] > self.max_image_size:
                self.max_image_size = params['image_size']
            self.emulators[key] = BaseEmulator(model_pths[i], params, device=self.device, norm_func=norm_funcs[i])
        #coordinates of the emulated region
        x = np.linspace(-3, 3, self.max_image_size)
        y = np.linspace(-3, 3, self.max_image_size)
        xx, yy = np.meshgrid(x, y)
        rr = hypot_func(xx, yy)
        pp = np.arctan2(yy, xx)
        self.dom_mask = (rr > 0.4) & (rr < 3)
        self.rr_dom = rr[self.dom_mask]
        self.pp_dom = pp[self.dom_mask]
        

    def emulate(self, alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['dens', 'vphi', 'vr'], norm=True, v_sign=+1):
        '''
            This returns the emulated fields of a disk in the cartesian coordinates defined in init.
        '''
        params_l = np.stack([planetMass, h, alpha, flaringIndex]).reshape(4,-1).T
        
        if norm:
            norm_params = norm_labels(params_l)
        else:
            norm_params = params_l
            sigmaSlope = (sigmaSlope+1)*(1.2-0.5)/2 + 0.5
        
        return self.emulate_normparams(norm_params=norm_params, sigmaSlope=sigmaSlope, fields=fields, v_sign=v_sign)
        
    
    def emulate_normparams(self, norm_params, sigmaSlope, fields=['dens', 'vphi', 'vr'], v_sign=+1):
        
        result = []
    
        ic = self.ict_gen(
            slopes=np.array([sigmaSlope]), dimension=self.max_image_size
        )
        
        for i, key in enumerate(fields):
            if self.emulators[key].params['image_size'] < self.max_image_size:
                #TODO: implement interpolation to smaller size. For now just use the same size for all fields.
                raise NotImplementedError()
            factor=1.
            if key=='vphi':
                factor=v_sign
            result_sing = factor*self.emulators[key](ic[:, [self.ict_comp_dict[key]]], norm_params).detach()
            if v_sign == +1:
                result_sing = result_sing.flip(-2)
            result.append(result_sing)
                
        return torch.concatenate(result, axis=1) #(N, fields, NX, NY)


    def emulate_dens(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        return self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['dens'])

    def emulate_vphi(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        return self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['vphi'])

    def emulate_vr(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        return self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['vr'])

    def per_b(self, t):
        shape = t.shape
        t = t.flatten()
        t[t > np.pi] = t[t > np.pi] - 2 * np.pi
        t[t < -np.pi] = t[t < -np.pi] + 2 * np.pi
        return t.reshape(*shape)

    def emulate_v3d(
        self,
        coord,
        alpha,
        h,
        planetMass,
        flaringIndex,
        R_p,
        phi_p,
        extrap_vfunc,
        sigmaSlope=None,
        norm=True,
        interp_3d='SPHERICAL',
        discminer_integr = True,
        mask_only_ppos = False,
        **extrap_kwargs,
    ):

        #prepare call to the extrapolation function
        for key, obj in extrap_kwargs.items():
            if callable(obj):
                extrap_kwargs[key] = obj(alpha=alpha,
                                         h=h,
                                         planetMass=planetMass,
                                         flaringIndex=flaringIndex,
                                         R_p=R_p,
                                         phi_p=phi_p,
                                         **extrap_kwargs)

        G = 6.67384e-11
        if "Mstar" in extrap_kwargs.keys():
            Mstar = extrap_kwargs["Mstar"]
            #print(f"using star mass Mstar={Mstar} Msun")
        else:
            #print("using default star mass Mstar=1 Msun")
            Mstar = 1

        #v_sign=+1 is clockwise
        if "vel_sign" in extrap_kwargs.keys():
            vel_sign = extrap_kwargs["vel_sign"]
        else:
            vel_sign = 1

        if sigmaSlope==None:
            if norm:
                #print('using norm=True')
                sigmaSlope = 2*np.array(flaringIndex) + 0.5
            else:
                #print('using norm=False')
                sigmaSlope = np.array(flaringIndex)
            
        v3d = (
            self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex, fields=['vphi', 'vr'], norm=norm, v_sign=vel_sign)
            .detach()
            .numpy()
        )

        rr_dom = self.rr_dom * R_p
        pp_dom = self.per_b(self.pp_dom + phi_p)
        v3d_dom = v3d[:,:, self.dom_mask]
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

        vphi_interp = np.array([
            (
                interpolator(
                    v3d_dom[i,0].reshape(-1)
                )
                * np.sqrt(G * Mstar * u.MSun / R_p)
            )
            * 1e-3 #this is because we use km
            
        for i in range(v3d_dom.shape[0])])
        
        
        vr_interp = np.array([(
            interpolator(v3d_dom[i,1].reshape(-1))
            * 1e-3 * np.sqrt(G * Mstar * u.MSun / R_p)
        )
                              for i in range(v3d_dom.shape[0])])

        mask = (R > 2.9 * R_p) | (R < 0.5 * R_p)

        mask_ppos = (R > 0.9* R_p) & (R < 1.1* R_p) & (phi<phi_p+0.5) & (phi>phi_p-0.5)
        
        vphi_interp[:,mask] = extrap_vfunc(coord, **extrap_kwargs)[mask]
        vr_interp[:,mask] = 0
        v3d_interp = np.concatenate(
            [
                np.expand_dims(vphi_interp, axis=1),
                np.expand_dims(vr_interp, axis=1),
                np.expand_dims(np.zeros(vphi_interp.shape), axis=1),
            ],
            axis=1,
        )

        if mask_only_ppos:
            v3d_interp[:,:,~mask_ppos] = np.nan

        if discminer_integr:
            v3d_interp = v3d_interp[0]


        return v3d_interp
