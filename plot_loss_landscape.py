
import numpy as np
import pandas as pd
import utils
import numpy as np

COLORS = {
    'stad': 'maroon',
    'scorebase': 'darkorange',
    'numeric': 'forestgreen',
    'score': 'steelblue'
}

def plot_loss_landscape_primal(ax, numeric_fit, par_vals, primal_list_m, primal_list,score_m, score_list, score_baseline_m, stad_m, numeric_m):
    Nscan,NMC = primal_list.shape
    print(primal_list.shape)

    ax.scatter(np.tile(np.transpose(par_vals.reshape(1,par_vals.shape[0])), (1,20)), 
                primal_list[:,:20],# + np.random.normal(0,0.01, size = score_list.shape), 
                alpha =  0.1, label="per event primal", c = 'k')

    # ax.scatter(par_vals, primal_list_m, label = 'primal mean')

    mean, mean_q = array_mean_and_quantiles(primal_list)

    ax.fill_between(par_vals,mean_q[0],mean_q[2], facecolor = 'steelblue', alpha = 0.5)
    ax.plot(par_vals,mean, label = 'primal median', c = 'steelblue')

    ax.plot(par_vals, numeric_fit(par_vals), color='k', label = 'poly. fit', linestyle = 'dashed')
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

    stad_mean, stad_q = utils.array_mean_and_quantiles(stad_list)
    scob_mean, scob_q = utils.array_mean_and_quantiles(score_baseline_list)
    scor_mean, scor_q = utils.array_mean_and_quantiles(score_list)
    numr_mean, numr_q = utils.array_mean_and_quantiles(numeric_list)

    ax.plot(par_vals, numr_mean, label = 'Numeric', color = COLORS['numeric'])
    ax.fill_between(par_vals,numr_q[0],numr_q[2], alpha = 0.2, facecolor = COLORS['numeric'])

    ax.plot(par_vals, stad_mean, label = 'STAD', color = COLORS['stad'])
    ax.fill_between(par_vals,stad_q[0],stad_q[2], alpha = 0.2, facecolor = COLORS['stad'])

    ax.plot(par_vals, scob_mean, label = 'SCOB', color = COLORS['scorebase'])
    ax.fill_between(par_vals,scob_q[0],scob_q[2], alpha = 0.2, facecolor = COLORS['scorebase'])

    ax.plot(par_vals, scor_mean, label = 'SCORE', color = COLORS['score'])
    ax.fill_between(par_vals,scor_q[0],scor_q[2], alpha = 0.2, facecolor = COLORS['score'])

    ax.plot(
        par_vals, grad_from_fit(par_vals), color='black', linestyle = 'dashed',
        label = 'grad from fit'
    )


    ax.set_xlim(0.0,4.0)
    ax.set_ylim(-15.0, 20.0)

    ax.set_ylabel('Grad')
    ax.set_xlabel('parameter')
    ax.legend(loc='upper left')
