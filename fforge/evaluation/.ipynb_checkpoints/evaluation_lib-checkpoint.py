import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

rangelen = {'PlanetMass': 3, 'AspectRatio': 0.07, 'Alpha': 2}
params = ['PlanetMass', 'AspectRatio', 'Alpha']
grids = {'Alpha': np.logspace(-4,-2,20),
                 'PlanetMass': np.logspace(-5,-2,20),
                 'AspectRatio': np.linspace(0.03, 0.1,20)}


'''
Returns the best estimate and errors for each parameter given the emulations and simulations (or the mse).
Old version
'''
def get_best(emulations, testset, sigma, mse={}, grids=grids):
    bestest = {}
    sigmas = {}
    valmins = {}
    errs = {}
    
    if len(mse)==0:
        print('Computing things...')
        for para in params:
            mse[para] = ((emulations[para]-np.expand_dims(testset, axis=1))**2/(2*sigma**2)).sum(axis=(-1,-2,-3))

    for para in tqdm(['PlanetMass', 'AspectRatio', 'Alpha']):
        M = len(grids[para])-1
        sigmas_l = []
        sigmas_r = []
        for i in range(mse[para].shape[0]):
            mses = mse[para][i]
            minimum = np.argmin(mses)
            N = np.min(mses)
            rm = minimum+1 if minimum<M else M
            lm = minimum-1 if minimum>0 else 0
            sigmas_l.append((1/((mses[lm]-2*mses[minimum]+mses[lm])/((rangelen[para])/100)**2/N))**0.5)
            sigmas_r.append((1/((mses[rm]-2*mses[minimum]+mses[rm])/((rangelen[para])/100)**2/N))**0.5)
        sigmas[para] = np.array([sigmas_l, sigmas_r])
        mins = np.argmin(mse[para], axis=1)
        bestest[para] = grids[para][mins]
        valmins[para] = np.min(mse[para], axis=1)
        if para=='AspectRatio':
            errs[para] = (-grids[para][mins]+sigmas[para][0]+grids[para][mins], grids[para][mins]+sigmas[para][1]-grids[para][mins])
        else:
            errs[para] = (-10**(np.log10(grids[para][mins])-sigmas[para][0])+grids[para][mins],
                          10**(np.log10(grids[para][mins])+sigmas[para][1])-grids[para][mins])    
        
    return bestest, errs, sigmas

'''
Returns the best estimate and errors for each parameter given the emulations and simulations (or the mse).
Uses a gaussian likelihood computed from the mse (which should be already normalized by sigma if provided as an argument)
Returns best estimate and uncertainties based on the 16th, 50th and 84th percentiles of the likelihood.
'''
def get_best_likeeqpost(emulations, testset, sigma, mse={}, grids=grids):
    bestest = {}
    sigmas = {}
    logsigma = {}
    valmins = {}
    errs = {}
    like = {}
    perc = {}

    if len(mse)==0:
        print('Computing things...')
        for para in params:
            mse[para] = ((emulations[para]-np.expand_dims(testset, axis=1))**2/(2*sigma**2)).sum(axis=(-1,-2,-3))

    for para in params:
        like[para] = np.exp(-1*(mse[para]-np.min(mse[para], axis=1).reshape(-1,1)))
        perc[para] = np.cumulative_sum(like[para], axis=-1)/np.sum(like[para], axis=-1).reshape(-1,1)        
        
    for para in tqdm(['PlanetMass', 'AspectRatio', 'Alpha']):
        sigmas_l = []
        sigmas_r = []
        bestest[para] = []
        for i in range(mse[para].shape[0]):
            sl, med, sr = np.interp([0.16,0.5, 0.84], perc[para][i], grids[para])
            sigmas_l.append(med-sl)
            sigmas_r.append(sr-med)
            bestest[para].append(med)
        sigmas[para] = np.array([sigmas_l, sigmas_r])
        bestest[para] = np.array(bestest[para])

        if para != 'AspectRatio':
            logsigma[para] = np.array([np.log10(bestest[para])-np.log10(bestest[para]-sigmas[para][0]),
                  np.log10(bestest[para]+sigmas[para][1])-np.log10(bestest[para])])
        else:
            logsigma[para] = sigmas[para]
        
    return bestest, sigmas, logsigma
    

def generate(emulations,
             testset,
             testparams,
             rangelen=rangelen,
             grids=grids,
             params=params,
             sigma=1/np.sqrt(2),
             savefolder='plots/',
             overwrite=False):

    if os.path.exists(savefolder):
        if overwrite:
            print('Saving folder exists, overwriting...')
        else:
            print('Saving folder exists. Run with overwrite=True if you want to overwrite the content')
            return
    else:
        os.mkdir(savefolder)

        
    bestest, errs, sigmas = get_best_likeeqpost(emulations, testset, sigma, mse={})

    #three scatter plots
    print('Generating scatter plots')
    fig, axs = plt.subplots(1, 3, figsize=(10,3))
    lims = {'Alpha': (1e-4, 1e-2), 'AspectRatio': (0.03, 0.1), 'PlanetMass': (1e-5,1e-2)}
    labels = {'PlanetMass': '$M_p/M_\star$', 'AspectRatio': 'h', 'Alpha': '$\\alpha$'}
    for i, para in enumerate(sigmas.keys()):
        axs[i].errorbar(testparams[para], 
                        bestest[para],
                        yerr=errs[para],
                        fmt='.',
                        color='black',
                        zorder=-1,
                       alpha=0.2)
        axs[i].set_rasterization_zorder(0)
        if i!=1:
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
        axs[i].set_xlim(*lims[para])
        axs[i].set_ylim(*lims[para])
        axs[i].plot(lims[para], lims[para], color='red', linewidth=0.5, linestyle='dashed')
        axs[i].set_xlabel(f'target {labels[para]}')
        axs[i].set_ylabel(f'{labels[para]} of minimum mse')
    
    plt.tight_layout()
    plt.savefig(f'{savefolder}/varp_all.pdf', dpi=500, bbox_inches='tight')
    plt.close()

    #single scatter plot with thermal mass
    tmass_un = 3*testparams['PlanetMass']/(2*testparams['AspectRatio']**3)
    para = 'PlanetMass'
    plt.errorbar(3*testparams[para]/(2*testparams['AspectRatio']**3), 
                        3*bestest[para]/(2*testparams['AspectRatio']**3),
                        yerr=3*np.array(errs[para])/(2*np.array(testparams['AspectRatio']).reshape(1,-1)**3),
                        fmt='.',
                        color='black',
                        zorder=-1,
                       alpha=0.2)
    plt.gca().set_rasterization_zorder(0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-2,300)
    plt.ylim(1e-2, 300)
    plt.plot(lims[para], lims[para], color='red', linewidth=0.5, linestyle='dashed')
    plt.xlabel('target $M_p/M_\\text{th}$')
    plt.ylabel('$M_p/M_\\text{th}$ of minimum mse')
    plt.plot([1e-2,300], [1e-2,300], color='red', linestyle='dashed', linewidth=1)
    plt.gca().set_aspect('equal')
    plt.savefig(f'{savefolder}/tmass_scatter.pdf', dpi=500, bbox_inches='tight')
    plt.close()
    
    #z plot
    print('Generating z plot')
    from scipy.stats import norm
    colors = {'Alpha': '#D81B60', 'PlanetMass': '#1E88E5', 'AspectRatio': '#004D40'}
    
    for para in sigmas.keys():
        nsigmas = sigmas[para][1]
        nsigmas[bestest[para]>testparams[para]] = sigmas[para][0][bestest[para]>testparams[para]]
        
        if para!='AspectRatio':
            errsigma = np.abs(np.log10(testparams[para])-np.log10(bestest[para]))/nsigmas
        else:
            errsigma = np.abs(testparams[para]-bestest[para])/nsigmas
        ecp, bins = np.histogram(errsigma, bins=np.linspace(0,5,50))
        ecp = np.cumsum(ecp)
        p = (norm.cdf(bins[1:])-0.5)*2
        plt.plot(bins[1:], ecp, linewidth=2, color=colors[para], label=labels[para])
    
    plt.xlabel('z')
    plt.ylabel('# of estimates with error < z')
    #plt.xlim(10,233)
    #plt.ylim(10,233)
    plt.plot(bins[1:], len(errsigma)*p, color='black', linestyle='dashed', alpha=0.3, label='Gaussian distribution')
    plt.fill_between(bins[1:],  p*len(errsigma)-(len(errsigma)*p*(1-p))**0.5,   p*len(errsigma)+(len(errsigma)*p*(1-p))**0.5, linewidth=2, color='black', alpha=0.1)
    #plt.gca().set_aspect('equal')
    plt.legend()
    plt.xscale('log')
    plt.xticks(range(1,6), range(1,6))
    plt.gcf().set_size_inches(5,4)
    plt.savefig(f'{savefolder}/zplot_simu.pdf', dpi=500, bbox_inches='tight')
    plt.close()

    return bestest
    
    
        