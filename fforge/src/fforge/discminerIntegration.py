from discminer.disc2d import Model
import numpy as np
import numbers
from discminer.diff_interp import get_griddata_sparse as get_griddata
import copy
from discminer.disc2d import *


class customDiscminerModel(Model):

    def __init_extra__(self, *args, **kwargs):
        self.first_makemodel = True

    def make_model(self, z_mirror=False, **kwargs_line_profile):  

        if self.first_makemodel:
            self.first_makemodel = False
            if self.prototype and self.verbose: 
                break_line()
                print ('Running prototype model with the following parameters:\n')
                pprint.pprint(self.params)
                break_line(init='\n')

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

            grid_true = {'upper': [self.x_true, self.y_true, z_true, self.R_true, self.phi_true, self.R_1d, z_1d], 
                        'lower': [self.x_true, self.y_true, z_true_far, self.R_true, self.phi_true, self.R_1d, z_far_1d]}
            self.grid_true = grid_true
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
            


def custom_ln_likelihood(self, new_params, **kwargs):

        for i in range(self.mc_nparams):
            if not (self.mc_boundaries_list[i][0] < new_params[i] < self.mc_boundaries_list[i][1]): return -np.inf
            else: self.params[self.mc_kind[i]][self.mc_header[i]] = new_params[i]
            
        vel2d, int2d, linew2d, lineb2d = self.make_model(**kwargs)

        lnx2=0    
        model_cube = self.get_cube(self.mc_vchannels, vel2d, int2d, linew2d, lineb2d, return_data_only=True)
        for i in range(self.mc_nchan):
            model_chan = model_cube[i]
            mask_data = np.isfinite(self.mc_data[i])
            mask_model = np.isfinite(model_chan)
            data = np.where(np.logical_and(mask_model, ~mask_data), 0, self.mc_data[i])
            model = np.where(np.logical_and(mask_data, ~mask_model), 0, model_chan)
            mask = np.logical_and(mask_data, mask_model)
            lnx =  np.where(mask, np.power((data - model),2) / (np.power(self.noise_stddev, 2)+ np.power(10, 2*self.params['likelihood']['log_emu_unc'])) , 0) 
            lnx2 += -0.5 * np.sum(lnx)
            
        return lnx2 if np.isfinite(lnx2) else -np.inf
    