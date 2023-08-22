
import numpy as np
import pandas as pd
import utils
import numpy as np


def plot_loss_landscape_primal(ax, numeric_fit, par_vals, primal_list):
    Nscan,NMC = primal_list.shape
    print(primal_list.shape)

    ax.scatter(np.tile(np.transpose(par_vals.reshape(1,par_vals.shape[0])), (1,20)), 
                primal_list[:,:20],# + np.random.normal(0,0.01, size = score_list.shape), 
                alpha =  0.1, label="per event primal", c = 'k')

    # ax.scatter(par_vals, primal_list_m, label = 'primal mean')

    mean, mean_q = utils.array_mean_and_quantiles(primal_list, window = 1)

    ax.plot(par_vals,np.array([mean_q[0],mean_q[2]]).T, color = 'k', alpha = 0.5)
    ax.plot(par_vals,mean, label = 'primal median', c = 'k')

    ax.plot(par_vals, numeric_fit(par_vals), color='maroon', label = 'poly. fit', linestyle = 'dashed')
    ax.axhline(0.0, c = 'k')
    ax.set_xlim(0.0,4.1)
    ax.set_ylim(-0.2,10)
    ax.set_ylabel('Loss')
    ax.set_xlabel('parameter')
    ax.legend()

def plot_loss_landscape_gradients(
    ax,
    numeric_fit,
    par_vals,
    numeric_list,
    score_list,
    score_baseline_list,
    stad_list):
    window = 3

    grad_from_fit = numeric_fit.deriv()

    w = 1
    stad_mean, stad_q = utils.array_mean_and_quantiles(stad_list, window = w)
    scob_mean, scob_q = utils.array_mean_and_quantiles(score_baseline_list, window = w)
    scor_mean, scor_q = utils.array_mean_and_quantiles(score_list, window = w)
    numr_mean, numr_q = utils.array_mean_and_quantiles(numeric_list, window = w)

    ax.plot(par_vals, numr_mean, label = 'Numeric', color = utils.COLORS['numeric'], linestyle = 'dashed')
    ax.plot(par_vals,np.array([numr_q[0],numr_q[2]]).T, alpha = 1.0, color = utils.COLORS['numeric'])

    ax.plot(par_vals, stad_mean, label = 'StochAD', color = utils.COLORS['stad'], linestyle = 'dashed')
    ax.plot(par_vals,np.array([stad_q[0],stad_q[2]]).T, alpha = 1.0, color = utils.COLORS['stad'])

    ax.plot(par_vals, stad_mean, label = 'SCORB', color = utils.COLORS['scorebase'], linestyle = 'dashed')
    ax.plot(par_vals,np.array([scob_q[0],scob_q[2]]).T, alpha = 1.0, color = utils.COLORS['scorebase'])

    ax.plot(par_vals, scor_mean, label = 'SCORE', color = utils.COLORS['score'], linestyle = 'dashed')
    ax.plot(par_vals,np.array([scor_q[0],scor_q[2]]).T, alpha = 1.0, color = utils.COLORS['score'])


    ax.plot(
        par_vals, grad_from_fit(par_vals), color='black', linestyle = 'dashed',
        label = 'grad from fit'
    )


    ax.set_xlim(0.0,4.0)
    ax.set_ylim(-15.0, 20.0)

    ax.set_ylabel('Grad')
    ax.set_xlabel('parameter')
    ax.legend(loc='upper left')
