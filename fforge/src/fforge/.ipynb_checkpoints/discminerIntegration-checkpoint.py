from discminer.disc2d import Model
import numpy as np
import numbers
from discminer.diff_interp import get_griddata_sparse as get_griddata
import copy
from discminer.disc2d import *
from discminer.tools.utils import FrontendUtils
import time
import torch
from multiprocessing import Pool as Pool

__all__ = ['Model', 'Mcmc', 'Velocity', 'Intensity', 'Linewidth', 'Lineslope', 'ScaleHeight', 'SurfaceDensity', 'Temperature']

try: 
    import termtables
    found_termtables = True
except ImportError:
    print ("\n*** For nicer outputs we recommend installing 'termtables' by typing in terminal: pip install termtables ***")
    found_termtables = False

_break_line = FrontendUtils._break_line

SMALL_SIZE = 10
MEDIUM_SIZE = 15

def _init_worker(model):
    global _GLOBAL_MODEL
    _GLOBAL_MODEL = model

def _log_prob(theta, **kwargs):
    return _GLOBAL_MODEL.ln_likelihood(theta, **kwargs)

def _log_prob_SC(theta, **kwargs):
    return _GLOBAL_MODEL.ln_likelihood_SC(theta, **kwargs)

_likelihood_dict = {
    'standard': _log_prob,
    'space_corr': _log_prob_SC
}

class customDiscminerModel(Model):

    def __init_extra__(self, *args, **kwargs):
        self.first_makemodel = True
        if 'ln_likelihood' in kwargs.keys():
            self.likelihood_key = kwargs['ln_likelihood']
        else:
            self.likelihood_key = 'standard'

    def __getstate__(self):
        print('pickling model')
        self.first_makemodel = True
        state = self.__dict__.copy()
        state.pop('gridd_data', None)  # remove unpicklable interpolators
        return state

    def populate(self, data=None, vchannels=None, p0_mean=[],
                 z_mirror=False,
                 frac_stddev=0.1,
                 **kwargs_model): 
        """
        Optimise the discminer model parameters using an MCMC sampler.

        Parameters
        __________
        data : array_like with shape (nchan, nx, nx), optional
            Reference intensity data to be modelled. If not specified, discminer considers the 
            *data* attribute of the original datacube with which the model was initialised.
        
        vchannels : array_like with shape (nchan,), optional
            Reference velocity channels matching the velocity axis of the input data. If not specified, discminer
            considers the velocity channels of the datacube that was used to initialise the model.

        p0_mean : array_like with shape (npars,)
            Mean value of initial-guess parameters. These will be sampled assuming a normal distribution.

        frac_stddev : float or array_like with shape (npars,), optional
            Fraction of the parameter range extent that will be considered as the standard deviation of the normal distribution of initial-guess parameters.
        
        frac_stats : float
            Fraction of MCMC steps at the end of the parameter chains considered for the computation of best-fit parameters (Defaults to 0.2, i.e. 20).
       
        """        

        self.noise_stddev = 1.0

        _break_line = FrontendUtils._break_line
        if data is None and vchannels is None:
            self.mc_data = self.datacube.data
            self.mc_vchannels = self.vchannels
        elif data is not None and vchannels is not None:
            self.mc_data = data
            self.mc_vchannels = vchannels
        else:
            raise InputError((data, vchannels),
                             'Please specify both data AND vchannel slices you wish to consider for the MCMC sampling.')
            
        self.mc_nchan = len(vchannels)
            
        kwargs_model.update({'z_mirror': z_mirror})
        if z_mirror: 
            for key in self.mc_params['height_lower']: self.mc_params['height_lower'][key] = 'height_upper_mirror'
        self.mc_header, self.mc_kind, self.mc_nparams, self.mc_boundaries_list, self.mc_params_indices = Model._get_params2fit(self.mc_params, self.mc_boundaries)
        self.params = copy.deepcopy(self.mc_params)

        if isinstance(p0_mean, (list, tuple, np.ndarray)): 
            if len(p0_mean) != self.mc_nparams: raise InputError(p0_mean, 'Length of input p0_mean must be equal to the number of parameters to fit: %d'%self.mc_nparams)
            else: pass

        ndim = self.mc_nparams

        p0_stddev = [frac_stddev*(self.mc_boundaries_list[i][1] - self.mc_boundaries_list[i][0]) for i in range(self.mc_nparams)]

        _break_line()
        print ('Initialising MCMC routines with the following (%d) parameters:\n'%self.mc_nparams)
        if found_termtables:
            bound_left, bound_right = np.array(self.mc_boundaries_list).T
            tt_header = ['Attribute', 'Parameter', 'Mean initial guess', 'Par stddev', 'Lower bound', 'Upper bound']
            tt_data = np.array([self.mc_kind, self.mc_header, p0_mean, p0_stddev, bound_left, bound_right]).T
            termtables.print(
                tt_data,
                header=tt_header,
                style=termtables.styles.markdown,
                padding=(0, 1),
                alignment="lcllll"
                )
        else:
            print ('Parameter header set for mcmc model fitting:', self.mc_header)
            print ('Parameters to fit and fixed parameters:')
            pprint.pprint(self.mc_params)
            print ('Number of mc parameters:', self.mc_nparams)
            print ('Parameter attributes:', self.mc_kind)
            print ('Parameter boundaries:')
            pprint.pprint(self.mc_boundaries_list)
            print ('Mean for initial guess p0:', p0_mean)
            print ('p0 pars stddev:', p0_stddev)
        _break_line(init='\n', end='\n\n')
        self.ln_likelihood(p0_mean)


    def make_model(self, z_mirror=False, **kwargs_line_profile):  
        _break_line = FrontendUtils._break_line

        if self.first_makemodel:
            self.first_makemodel = False
            if self.prototype and self.verbose: 
                _break_line()
                print ('Running prototype model with the following parameters:\n')
                pprint.pprint(self.params)
                _break_line(init='\n')

            incl, PA, xc, yc = self.orientation_func({'R': self.R_true}, **self.params['orientation'])
            int_kwargs = self.params['intensity']
            vel_kwargs = self.params['velocity']
            lw_kwargs = self.params['linewidth']
            ls_kwargs = self.params['lineslope']
            self.update_beam_from_params() #update beam info if relevant; convolution happens in get_cube or get_channel
        
            cos_incl, sin_incl = np.cos(incl), np.sin(incl)
            self.cos_incl = cos_incl
            self.sin_incl = sin_incl
        
            #*******************************************
            #MAKE TRUE GRID FOR UPPER AND LOWER SURFACES
            z_true = self.z_upper_func({'R': self.R_true, 'phi': self.phi_true}, **self.params['height_upper'])

            if z_mirror: z_true_far = -z_true
            else: z_true_far = self.z_lower_func({'R': self.R_true, 'phi': self.phi_true}, **self.params['height_lower']) 

            if (self.velocity_func is Velocity.keplerian_vertical_selfgravity or
                self.velocity_func is Velocity.keplerian_vertical_selfgravity_pressure):
                z_1d = self.z_upper_func({'R': self.R_1d*sfu.au}, **self.params['height_upper'])/sfu.au
                if z_mirror: z_far_1d = -z_1d
                else: z_far_1d = self.z_lower_func({'R': self.R_1d*sfu.au}, **self.params['height_lower'])/sfu.au
            else: z_1d = z_far_1d = None

            self.grid_true = {'upper': [self.x_true, self.y_true, z_true, self.R_true, self.phi_true, self.R_1d, z_1d], 
                        'lower': [self.x_true, self.y_true, z_true_far, self.R_true, self.phi_true, self.R_1d, z_far_1d]}
            grid_true = self.grid_true
            #******************************
            #COMPUTE PROPERTIES ON SKY GRID 
            avai_kwargs = [vel_kwargs, int_kwargs, lw_kwargs, ls_kwargs]
            avai_funcs = [self.velocity_func, self.intensity_func, self.linewidth_func, self.lineslope_func]
            true_kwargs = [isinstance(kwarg, dict) for kwarg in avai_kwargs]
            prop_kwargs = [kwarg for i, kwarg in enumerate(avai_kwargs) if true_kwargs[i]]
            prop_funcs = [func for i, func in enumerate(avai_funcs) if true_kwargs[i]]
            nfuncs = len(prop_funcs)
            
            if self.subpixels:

                props = [[] for k in range(nfuncs)]

                for i in range(self.subpixels):
                    for j in range(self.subpixels):
                        z_true = self.z_upper_func({'R': self.sub_R_true[i][j]}, **self.params['height_upper'])
                        
                        if z_mirror: z_true_far = -z_true
                        else: z_true_far = self.z_lower_func({'R': self.sub_R_true[i][j]}, **self.params['height_lower']) 

                        subpix_grid_true = {'upper': [self.sub_x_true[j], self.sub_y_true[i], z_true, self.sub_R_true[i][j], self.sub_phi_true[i][j], None, None], 
                                            'lower': [self.sub_x_true[j], self.sub_y_true[i], z_true_far, self.sub_R_true[i][j], self.sub_phi_true[i][j], None, None]}
                        #subpix_vel.append(self._compute_prop(subpix_grid_true, [self.velocity_func], [vel_kwargs])[0])
                        for k in range(nfuncs):
                            tmp = self._compute_prop(subpix_grid_true, [prop_funcs[k]], [prop_kwargs[k]])[0]
                            if true_kwargs[0] and k==0: #i.e. velocity
                                ang_fac = sin_incl * np.cos(self.sub_phi_true[i][j])
                                for side in ['upper', 'lower']:
                                    tmp[side] *= ang_fac
                                    tmp[side] += vel_kwargs['vsys']
                                    
                            props[k].append(tmp)                    

            else: 
                props = self._compute_prop(grid_true, prop_funcs, prop_kwargs)
                if true_kwargs[0]: #Convention: positive vel (+) means gas receding from observer
                    phi_fac = sin_incl * np.cos(self.phi_true) #phi component
                    for side in ['upper', 'lower']:
                        if len(props[0][side])==3: #3D vel
                            v3d = props[0][side]
                            r_fac = sin_incl * np.sin(self.phi_true)
                            z_fac = cos_incl
                            props[0][side] = v3d[0]*phi_fac - v3d[1]*r_fac - v3d[2]*z_fac
                        else: #1D vel, assuming vphi only
                            props[0][side] *= phi_fac 
                        props[0][side] += vel_kwargs['vsys']

            #***********************************
            #PROJECT PROPERTIES ON THE SKY PLANE 
            self.gridd_data = {}       
            x_pro_dict = {}
            y_pro_dict = {}
            z_pro_dict = {}
            for side in ['upper', 'lower']:
                xt, yt, zt = grid_true[side][:3]
                x_pro, y_pro, z_pro = self._project_on_skyplane(xt, yt, zt, cos_incl, sin_incl)
                if len(np.atleast_1d(PA)) > 0:
                    x_pro, y_pro = self._rotate_sky_plane_ewise(x_pro, y_pro, PA)
                else:
                    if PA != 0.0:
                        x_pro, y_pro = self._rotate_sky_plane(x_pro, y_pro, PA)                    
                x_pro = x_pro+xc
                y_pro = y_pro+yc
                
                self.gridd_data[side] = get_griddata((x_pro, y_pro), (self.mesh[0], self.mesh[1]))
                if self.Rmax_m is not None:
                    R_grid = self.gridd_data[side](self.R_true)
                    self.R_grid = R_grid #griddata((x_pro, y_pro), self.R_true, (self.mesh[0], self.mesh[1]), method='linear')

                x_pro_dict[side] = x_pro
                y_pro_dict[side] = y_pro
                z_pro_dict[side] = z_pro

                if self.subpixels:

                    for prop in props:
                        for i in range(self.subpixels_sq): #Subpixels are projected on the same plane where true grid is projected
                            if not isinstance(prop[i][side], numbers.Number):
                                prop[i][side] =  self.gridd_data[side](prop[i][side]) #griddata((x_pro, y_pro), prop[i][side], (self.mesh[0], self.mesh[1]), method='linear')
                            if self.Rmax_m is not None:
                                prop[i][side] = np.where(np.logical_and(R_grid<self.Rmax_m, R_grid>self.Rmin_m), prop[i][side], np.nan)

                else:
                    for prop in props:
                        if not isinstance(prop[side], numbers.Number): prop[side] = self.gridd_data[side](prop[side]) #griddata((x_pro, y_pro), prop[side], (self.mesh[0], self.mesh[1]), method='linear')
                        if self.Rmax_m is not None: prop[side] = np.where(np.logical_and(R_grid<self.Rmax_m, R_grid>self.Rmin_m), prop[side], np.nan)

            #*************************************
            if self.prototype:
                self.get_projected_coords(z_mirror=z_mirror) #TODO: enable kwargs for this method
                self.props = props
                return self.get_cube(self.vchannels, *props, header=self.header, dpc=self.dpc, disc=self.datacube.disc, mol=self.datacube.mol, kind=self.datacube.kind, **kwargs_line_profile)
            else:
                return props
        else:
            #print('make model')
            int_kwargs = self.params['intensity']
            vel_kwargs = self.params['velocity']
            lw_kwargs = self.params['linewidth']
            ls_kwargs = self.params['lineslope']
            #******************************
            #COMPUTE PROPERTIES ON SKY GRID 
            avai_kwargs = [vel_kwargs, int_kwargs, lw_kwargs, ls_kwargs]
            avai_funcs = [self.velocity_func, self.intensity_func, self.linewidth_func, self.lineslope_func]
            true_kwargs = [isinstance(kwarg, dict) for kwarg in avai_kwargs]
            prop_kwargs = [kwarg for i, kwarg in enumerate(avai_kwargs) if true_kwargs[i]]
            prop_funcs = [func for i, func in enumerate(avai_funcs) if true_kwargs[i]]
            nfuncs = len(prop_funcs)
            
            if self.subpixels:

                props = [[] for k in range(nfuncs)]

                for i in range(self.subpixels):
                    for j in range(self.subpixels):
                        z_true = self.z_upper_func({'R': self.sub_R_true[i][j]}, **self.params['height_upper'])
                        
                        if z_mirror: z_true_far = -z_true
                        else: z_true_far = self.z_lower_func({'R': self.sub_R_true[i][j]}, **self.params['height_lower']) 

                        subpix_grid_true = {'upper': [self.sub_x_true[j], self.sub_y_true[i], z_true, self.sub_R_true[i][j], self.sub_phi_true[i][j], None, None], 
                                            'lower': [self.sub_x_true[j], self.sub_y_true[i], z_true_far, self.sub_R_true[i][j], self.sub_phi_true[i][j], None, None]}
                        #subpix_vel.append(self._compute_prop(subpix_grid_true, [self.velocity_func], [vel_kwargs])[0])
                        for k in range(nfuncs):
                            tmp = self._compute_prop(subpix_grid_true, [prop_funcs[k]], [prop_kwargs[k]])[0]
                            if true_kwargs[0] and k==0: #i.e. velocity
                                ang_fac = self.sin_incl * np.cos(self.sub_phi_true[i][j])
                                for side in ['upper', 'lower']:
                                    tmp[side] *= ang_fac
                                    tmp[side] += vel_kwargs['vsys']
                                    
                            props[k].append(tmp)                    

            else: 
                props = self._compute_prop(self.grid_true, prop_funcs, prop_kwargs)
                if true_kwargs[0]: #Convention: positive vel (+) means gas receding from observer
                    phi_fac = self.sin_incl * np.cos(self.phi_true) #phi component
                    for side in ['upper', 'lower']:
                        if len(props[0][side])==3: #3D vel
                            v3d = props[0][side]
                            r_fac = self.sin_incl * np.sin(self.phi_true)
                            z_fac = self.cos_incl
                            props[0][side] = v3d[0]*phi_fac - v3d[1]*r_fac - v3d[2]*z_fac
                        else: #1D vel, assuming vphi only
                            props[0][side] *= phi_fac 
                        props[0][side] += vel_kwargs['vsys']

            #***********************************
            #PROJECT PROPERTIES ON THE SKY PLANE        
            for side in ['upper', 'lower']:
                if self.subpixels:
                    for prop in props:
                        for i in range(self.subpixels_sq): #Subpixels are projected on the same plane where true grid is projected
                            if not isinstance(prop[i][side], numbers.Number):
                                prop[i][side] =  self.gridd_data[side](prop[i][side]) #griddata((x_pro, y_pro), prop[i][side], (self.mesh[0], self.mesh[1]), method='linear')
                            if self.Rmax_m is not None:
                                prop[i][side] = np.where(np.logical_and(self.R_grid<self.Rmax_m, self.R_grid>self.Rmin_m), prop[i][side], np.nan)

                else:
                    for prop in props:
                        if not isinstance(prop[side], numbers.Number): prop[side] = self.gridd_data[side](prop[side]) #griddata((x_pro, y_pro), prop[side], (self.mesh[0], self.mesh[1]), method='linear')
                        if self.Rmax_m is not None: prop[side] = np.where(np.logical_and(self.R_grid<self.Rmax_m, self.R_grid>self.Rmin_m), prop[side], np.nan)

        #*************************************
            if self.prototype:
                self.get_projected_coords(z_mirror=z_mirror) #TODO: enable kwargs for this method
                self.props = props
                return self.get_cube(self.vchannels, *props, header=self.header, dpc=self.dpc, disc=self.datacube.disc, mol=self.datacube.mol, kind=self.datacube.kind, **kwargs_line_profile)
            else:
                return props
            


    def ln_likelihood_fixed_model_unc(self, new_params, **kwargs):
            #t1 = time.time()
            torch.set_num_threads(1)
            for i in range(self.mc_nparams):
                if not (self.mc_boundaries_list[i][0] < new_params[i] < self.mc_boundaries_list[i][1]): return -np.inf
                else: self.params[self.mc_kind[i]][self.mc_header[i]] = new_params[i]
            #t2 = time.time()
            #print(f'preparation: {t2-t1:.3f}')
            vel2d, int2d, linew2d, lineb2d = self.make_model(**kwargs)
            #t3 = time.time()
            #print(f'make model:  {t3-t2:.3f}')
            sigma_data_sq = np.power(self.noise_stddev, 2)
            sigma_emu_sq  = np.power(10, (2 * self.params['likelihood']['log_emu_unc']))
            sigma_tot_sq  = sigma_data_sq + sigma_emu_sq  # scalar, same for all pixels

            log_norm = np.log(2 * np.pi * sigma_tot_sq)
            lnx2=0    
            model_cube = self.get_cube(self.mc_vchannels, vel2d, int2d, linew2d, lineb2d, return_data_only=True)
            #t4 = time.time()
            #print(f'model cube: {t4-t3:.3f}')
            for i in range(self.mc_nchan):
                model_chan = model_cube[i]
                mask_data = np.isfinite(self.mc_data[i])
                mask_model = np.isfinite(model_chan)
                data = np.where(np.logical_and(mask_model, ~mask_data), 0, self.mc_data[i])
                model = np.where(np.logical_and(mask_data, ~mask_model), 0, model_chan)
                mask = np.logical_and(mask_data, mask_model)
                
                resid_sq =  np.where(mask, np.power((data - model),2), 0.0)
                n_valid = np.count_nonzero(mask)
                
                lnx2 += -0.5 * (np.sum(np.where(mask, resid_sq/sigma_tot_sq + log_norm, 0)))
                
            #t5 = time.time()
            #print(f'final things: {t5-t4:.3f}')
            return lnx2 if np.isfinite(lnx2) else -np.inf


    def ln_likelihood(self, new_params, **kwargs):
            #t1 = time.time()
            torch.set_num_threads(1)
            for i in range(self.mc_nparams):
                if not (self.mc_boundaries_list[i][0] < new_params[i] < self.mc_boundaries_list[i][1]): return -np.inf
                else: self.params[self.mc_kind[i]][self.mc_header[i]] = new_params[i]
            #t2 = time.time()
            #print(f'preparation: {t2-t1:.3f}')
            vel2d, int2d, linew2d, lineb2d = self.make_model(**kwargs)
            #t3 = time.time()
            #print(f'make model:  {t3-t2:.3f}')
            logplanetMass = (self.params['velocity']['planetMass']+1)/2*3-5
            sigma_data_sq = np.power(self.noise_stddev, 2)
            sigma_emu_sq  = np.power(10, (2 * (self.params['likelihood']['kappa']*logplanetMass+self.params['likelihood']['beta'])))
            sigma_tot_sq  = sigma_data_sq + sigma_emu_sq  # scalar, same for all pixels

            log_norm = np.log(2 * np.pi * sigma_tot_sq)
            lnx2=0    
            model_cube = self.get_cube(self.mc_vchannels, vel2d, int2d, linew2d, lineb2d, return_data_only=True)
            #t4 = time.time()
            #print(f'model cube: {t4-t3:.3f}')
            for i in range(self.mc_nchan):
                model_chan = model_cube[i]
                mask_data = np.isfinite(self.mc_data[i])
                mask_model = np.isfinite(model_chan)
                data = np.where(np.logical_and(mask_model, ~mask_data), 0, self.mc_data[i])
                model = np.where(np.logical_and(mask_data, ~mask_model), 0, model_chan)
                mask = np.logical_and(mask_data, mask_model)
                
                resid_sq =  np.where(mask, np.power((data - model),2), 0.0)
                n_valid = np.count_nonzero(mask)
                
                lnx2 += -0.5 * (np.sum(np.where(mask, resid_sq/sigma_tot_sq + log_norm, 0)))
                
            #t5 = time.time()
            #print(f'final things: {t5-t4:.3f}')
            return lnx2 if np.isfinite(lnx2) else -np.inf


    def ln_likelihood_SC(self, new_params, **kwargs):
            #t1 = time.time()
            torch.set_num_threads(1)
            for i in range(self.mc_nparams):
                if not (self.mc_boundaries_list[i][0] < new_params[i] < self.mc_boundaries_list[i][1]): return -np.inf
                else: self.params[self.mc_kind[i]][self.mc_header[i]] = new_params[i]
            #t2 = time.time()
            #print(f'preparation: {t2-t1:.3f}')
            vel2d, int2d, linew2d, lineb2d = self.make_model(**kwargs)
            #t3 = time.time()
            #print(f'make model:  {t3-t2:.3f}')
            logplanetMass = (self.params['velocity']['planetMass']+1)/2*3-5
            sigma_data_sq = np.power(self.noise_stddev, 2)
            sigma_emu_sq  = np.power(10, (2 * (self.params['likelihood']['kappa']*logplanetMass+self.params['likelihood']['beta'])))
            sigma_tot_sq  = sigma_data_sq + sigma_emu_sq  # scalar, same for all pixels

            log_norm = np.log(2 * np.pi * sigma_tot_sq)
            lnx2=0    
            model_cube = self.get_cube(self.mc_vchannels, vel2d, int2d, linew2d, lineb2d, return_data_only=True)
            #t4 = time.time()
            #print(f'model cube: {t4-t3:.3f}')
            for i in range(self.mc_nchan):
                model_chan = model_cube[i]
                mask_data = np.isfinite(self.mc_data[i])
                mask_model = np.isfinite(model_chan)
                data = np.where(np.logical_and(mask_model, ~mask_data), 0, self.mc_data[i])
                model = np.where(np.logical_and(mask_data, ~mask_model), 0, model_chan)
                mask = np.logical_and(mask_data, mask_model)
                
                resid_sq =  np.where(mask, np.power((data - model),2), 0.0)
                n_valid = np.count_nonzero(mask)
                
                lnx2 += -0.5 * (np.sum(np.where(mask, resid_sq/sigma_tot_sq + log_norm, 0)))
                
            #t5 = time.time()
            #print(f'final things: {t5-t4:.3f}')
            return lnx2 if np.isfinite(lnx2) else -np.inf

            
    def run_mcmc(self, data=None, vchannels=None, p0_mean=[], frac_stddev=1e-3,  
                 nwalkers=30, nsteps=100, frac_stats=0.2, noise_stddev=1.0,
                 nthreads=None,
                 backend=None, #emcee
                 use_zeus=False,
                 #custom_header={}, custom_kind={}, mc_layers=1,
                 z_mirror=False, 
                 plot_walkers=True,
                 plot_corner=True,
                 write_log_pars=True,
                 tag='',
                 mpi=False,
                 **kwargs_model): 
        """
        Optimise the discminer model parameters using an MCMC sampler.

        Parameters
        __________
        data : array_like with shape (nchan, nx, nx), optional
            Reference intensity data to be modelled. If not specified, discminer considers the 
            *data* attribute of the original datacube with which the model was initialised.
        
        vchannels : array_like with shape (nchan,), optional
            Reference velocity channels matching the velocity axis of the input data. If not specified, discminer
            considers the velocity channels of the datacube that was used to initialise the model.

        p0_mean : array_like with shape (npars,)
            Mean value of initial-guess parameters. These will be sampled assuming a normal distribution.

        frac_stddev : float or array_like with shape (npars,), optional
            Fraction of the parameter range extent that will be considered as the standard deviation of the normal distribution of initial-guess parameters.
        
        frac_stats : float
            Fraction of MCMC steps at the end of the parameter chains considered for the computation of best-fit parameters (Defaults to 0.2, i.e. 20).
       
        """        
        if data is None and vchannels is None:
            self.mc_data = self.datacube.data
            self.mc_vchannels = self.vchannels
        elif data is not None and vchannels is not None:
            self.mc_data = data
            self.mc_vchannels = vchannels
        else:
            raise InputError((data, vchannels),
                             'Please specify both data AND vchannel slices you wish to consider for the MCMC sampling.')
            
        self.mc_nchan = len(vchannels)
        self.noise_stddev = noise_stddev
        if use_zeus: import zeus as sampler_id
        else: import emcee as sampler_id
            
        kwargs_model.update({'z_mirror': z_mirror})
        if z_mirror: 
            for key in self.mc_params['height_lower']: self.mc_params['height_lower'][key] = 'height_upper_mirror'
        self.mc_header, self.mc_kind, self.mc_nparams, self.mc_boundaries_list, self.mc_params_indices = Model._get_params2fit(self.mc_params, self.mc_boundaries)
        self.params = copy.deepcopy(self.mc_params)

        if isinstance(p0_mean, (list, tuple, np.ndarray)): 
            if len(p0_mean) != self.mc_nparams: raise InputError(p0_mean, 'Length of input p0_mean must be equal to the number of parameters to fit: %d'%self.mc_nparams)
            else: pass

        nstats = int(round(frac_stats*(nsteps-1)))
        ndim = self.mc_nparams

        p0_stddev = [frac_stddev*(self.mc_boundaries_list[i][1] - self.mc_boundaries_list[i][0]) for i in range(self.mc_nparams)]
        p0 = np.random.normal(loc=p0_mean,
                              scale=p0_stddev,
                              size=(nwalkers, ndim)
                              )

        _break_line()
        print ('Initialising MCMC routines with the following (%d) parameters:\n'%self.mc_nparams)
        if found_termtables:
            bound_left, bound_right = np.array(self.mc_boundaries_list).T
            tt_header = ['Attribute', 'Parameter', 'Mean initial guess', 'Par stddev', 'Lower bound', 'Upper bound']
            tt_data = np.array([self.mc_kind, self.mc_header, p0_mean, p0_stddev, bound_left, bound_right]).T
            termtables.print(
                tt_data,
                header=tt_header,
                style=termtables.styles.markdown,
                padding=(0, 1),
                alignment="lcllll"
                )
        else:
            print ('Parameter header set for mcmc model fitting:', self.mc_header)
            print ('Parameters to fit and fixed parameters:')
            pprint.pprint(self.mc_params)
            print ('Number of mc parameters:', self.mc_nparams)
            print ('Parameter attributes:', self.mc_kind)
            print ('Parameter boundaries:')
            pprint.pprint(self.mc_boundaries_list)
            print ('Mean for initial guess p0:', p0_mean)
            print ('p0 pars stddev:', p0_stddev)
        _break_line(init='\n', end='\n\n')

        if mpi: #Needs schwimmbad library: $ pip install schwimmbad 
            from schwimmbad import MPIPool

            with MPIPool() as pool:
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)
                
                sampler = sampler_id.EnsembleSampler(nwalkers, ndim, _likelihood_dict[self.likelihood_key], pool=pool, backend=backend, kwargs=kwargs_model)                                                        
                start = time.time()
                if backend is not None and backend.iteration!=0:
                    sampler.run_mcmc(None, nsteps, progress=True)
                else:
                    sampler.run_mcmc(p0, nsteps, progress=True)
                end = time.time()
                multi_time = end - start
                print("MPI multiprocessing took {0:.1f} seconds".format(multi_time))

        else:
            with Pool(processes=nthreads, initializer=_init_worker, initargs=(self,)) as pool:
                sampler = sampler_id.EnsembleSampler(nwalkers, ndim, _log_prob, pool=pool, backend=backend, kwargs=kwargs_model)                                                      
                start = time.time()
                if backend is not None and backend.iteration!=0:
                    sampler.run_mcmc(None, nsteps, progress=True)
                else:
                    sampler.run_mcmc(p0, nsteps, progress=True)
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
            
        sampler_chain = sampler.chain
        if use_zeus: sampler_chain = np.swapaxes(sampler.chain, 0, 1) #zeus chains shape (nsteps, nwalkers, npars) must be swapped
        samples = sampler_chain[:, -nstats:] #3d matrix, chains shape (nwalkers, nstats, npars)
        samples = samples.reshape(-1, samples.shape[-1]) #2d matrix, shape (nwalkers*nstats, npars). With -1 numpy guesses the x dimensionality
        best_params = np.median(samples, axis=0)
        self.mc_sampler = sampler
        self.mc_samples = samples
        self.best_params = best_params

        samples_all = sampler_chain[:, :] #3d matrix, chains shape (nwalkers, nsteps, npars)
        samples_all = samples_all.reshape(-1, samples.shape[-1]) #2d matrix, shape (nwalkers*nsteps, npars)
        self.mc_samples_all = samples_all
        
        #Errors: +- 68.2 percentiles
        errpos, errneg = [], []
        for i in range(self.mc_nparams):
            tmp = best_params[i]
            indpos = samples[:,i] > tmp
            indneg = samples[:,i] < tmp
            val = samples[:,i][indpos] - tmp
            errpos.append(np.percentile(val, [68.2])) #1 sigma (2x perc 34.1), positive pars
            val = np.abs(samples[:,i][indneg] - tmp)
            errneg.append(np.percentile(val, [68.2])) 
        self.best_params_errpos = np.asarray(errpos).squeeze()
        self.best_params_errneg = np.asarray(errneg).squeeze()
        
        best_fit_dict = np.array([np.atleast_1d(arr) for arr in [p0_mean, best_params, self.best_params_errneg, self.best_params_errpos]]).T
        best_fit_dict = {key+'_'+self.mc_kind[i]: str(best_fit_dict[i].tolist())[1:-1] for i,key in enumerate(self.mc_header)}
        self.best_fit_dict = best_fit_dict
        
        _break_line(init='\n')
        print ('Median from parameter walkers for the last %d steps:\n'%nstats)        
        if found_termtables:
            tt_header = ['Parameter', 'Best-fit value', 'error [-]', 'error [+]']
            tt_data = np.array([np.atleast_1d(arr) for arr in [self.mc_header, self.best_params, self.best_params_errneg, self.best_params_errpos]]).T
            termtables.print(
                tt_data,
                header=tt_header,
                style=termtables.styles.markdown,
                padding=(0, 1),
                alignment="clll"
                )
        else:
            print (list(zip(self.mc_header, best_params)))
        _break_line(init='\n', end='\n\n')

        #************
        #PLOTTING
        #************
        #for key in custom_header: self.mc_header[key] = custom_header[key]
        #for key in custom_kind: self.mc_kind[key] = custom_kind[key]
        if plot_walkers: 
            Mcmc.plot_walkers(sampler_chain.T, best_params, header=self.mc_header, kind=self.mc_kind, nstats=nstats, tag=tag)
        if plot_corner: 
            Mcmc.plot_corner(samples, labels=self.mc_header)
            plt.savefig('mc_corner_%s_%dwalkers_%dsteps.png'%(tag, nwalkers, nsteps))
            plt.close()

        if write_log_pars:

            cp = lambda x: copy.deepcopy(x)

            params = cp(self.params) #Fit and fixed parameters
            p0pars = cp(self.params)
            errpos = cp(self.params)
            errneg = cp(self.params)

            #print (self.best_fit_dict)
            for key in self.best_fit_dict: 
                par = key.split('_')[0]
                attribute = key.split(par+'_')[1]
                val = self.best_fit_dict[key].split(',')

                p0pars[attribute][par] = float(val[0])
                params[attribute][par] = float(val[1])
                errneg[attribute][par] = float(val[2])        
                errpos[attribute][par] = float(val[3])    

            allheader = []
            allp0 = []
            allpars = []
            allerrpos = []
            allerrneg = [] 
    
            for attribute in params:
                for par in params[attribute]:
                    allheader.append(par)
                    allp0.append(p0pars[attribute][par])
                    allpars.append(params[attribute][par])
                    allerrneg.append(errneg[attribute][par])
                    allerrpos.append(errpos[attribute][par])
        
            #print (allheader, allpars)
            np.savetxt('log_pars_%s_cube_%dwalkers_%dsteps.txt'%(tag, nwalkers, backend.iteration),
                       np.array([allp0, allpars, allerrneg, allerrpos]), fmt='%.6f', header=str(allheader))


import numpy as np
from scipy.fft import fft2, ifft2

def get_chi2_fft_normalized(r, kernel, noise):
    """
    Computes the exact, fully normalized chi^2 spatial covariance penalty.
    """
    Ny, Nx = r.shape
    pad_shape = (2 * Ny, 2 * Nx) 
    
    # 1. Align phase center [0, 0]
    cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
    kernel_shifted = np.roll(kernel, shift=(-cy, -cx), axis=(0, 1))
    
    # 2. Forward FFT of the beam
    fkernel = fft2(kernel_shifted, s=pad_shape, workers=-1)
    
    # 3. Compute |F(B)|^2 
    # To fix the normalization perfectly regardless of physical pixel size units,
    # we force the discrete inverse-transform of the power spectrum to have a 
    # peak value (zero-lag variance) exactly equal to noise**2.
    power_spectrum = np.abs(fkernel)**2
    
    # Find what the discrete peak value would be in real-space
    # (The peak of a circular autocorrelation always lands at index [0,0])
    discrete_b_cov = np.real(ifft2(power_spectrum, workers=-1))
    discrete_peak = discrete_b_cov[0, 0]
    
    # Apply the exact normalization factor
    FBcov = (noise**2) * (power_spectrum / discrete_peak)
    
    # 4. Transform the residuals
    Fr = fft2(r, s=pad_shape, workers=-1)
    
    # 5. Element-wise division 
    Finv_Sigma_r = Fr / (FBcov + 1e-16)
    
    # 6. Bring back to pixel space and crop
    inv_Sigma_r_padded = np.real(ifft2(Finv_Sigma_r, workers=-1))
    inv_Sigma_r = inv_Sigma_r_padded[:Ny, :Nx]
    
    # 7. Compute the scalar chi2
    chi2 = np.sum(r * inv_Sigma_r)
    
    return chi2