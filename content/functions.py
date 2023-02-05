import numpy as np


def calc_metrics(sim,obs):
    outdict = {}

    N = obs.shape[0]
    outdict['N'] = N

    meansim = np.nanmean(sim)
    meanobs = np.nanmean(obs)
    sigmasim = np.nanstd(sim)
    sigmaobs = np.nanstd(obs)

    outdict['mean_DEHM'] = meansim
    outdict['mean_OBS'] = meanobs
    outdict['std_DEHM'] = sigmasim
    outdict['std_OBS'] = sigmaobs

    diff = sim - obs
    MB = np.nanmean(diff)
    outdict['MB'] = MB

    square_diff = np.square(diff)
    mean_square_diff = np.nanmean(square_diff)
    RMSE = np.sqrt(mean_square_diff)
    outdict['RMSE'] = RMSE

    addition = np.absolute(sim)+np.absolute(obs)
    division = np.where(addition==0, np.nan, np.true_divide(diff, addition))
    NMB = 2*np.nanmean(division)
    outdict['NMB'] = NMB
    
    diffsim = sim - meansim
    diffobs = obs - meanobs
    multidiff = np.multiply(diffsim, diffobs)
    CORR = np.nanmean(multidiff)/(sigmasim*sigmaobs)
    # print('For CORR calculations: ')
    outdict['COV'] = np.nanmean(multidiff)
    outdict['corr'] = CORR

    return outdict