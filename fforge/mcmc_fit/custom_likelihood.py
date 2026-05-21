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
    