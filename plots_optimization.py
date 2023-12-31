import pandas as pd
import utils
import numpy as np

def moving_avg(array, window):
    return pd.DataFrame(array).rolling(window).mean()

def plot_single_opt_comparison(ax, l_st, l_s, l_sb, l_n):
    window = 10
    ax.plot(moving_avg(l_st, window) , label = 'STAD', color = utils.COLORS['stad'])
    ax.plot(moving_avg(l_s, window), label = 'Score', color = utils.COLORS['score'])
    ax.plot(moving_avg(l_sb, window), label = 'Score Baseline', color = utils.COLORS['scorebase'])
    ax.plot(moving_avg(l_n, window), label = 'Numeric', color = utils.COLORS['numeric'])

    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylim(0,5)
    ax.legend()    


def plot_optimization_comparison(
    ax,
    l_st_list,
    l_s_list,
    l_n_list,
    l_sb_list
    
):

    st = utils.array_mean_and_quantiles(l_st_list.T)
    s = utils.array_mean_and_quantiles(l_s_list.T)
    n = utils.array_mean_and_quantiles(l_n_list.T)
    sb = utils.array_mean_and_quantiles(l_sb_list.T)

    xrange = np.arange(500)
    ax.plot(xrange,n[0],c = utils.COLORS['numeric'], label = 'numeric')
    ax.fill_between(xrange,n[1][0],n[1][2], facecolor = utils.COLORS['numeric'], alpha = 0.2)

    ax.plot(xrange,s[0],c = utils.COLORS['score'], label = 'SCORE')
    ax.fill_between(xrange,s[1][0],s[1][2], facecolor = utils.COLORS['score'], alpha = 0.2)

    ax.plot(xrange,sb[0],c = utils.COLORS['scorebase'], label = 'SCORB')
    ax.fill_between(xrange,sb[1][0],sb[1][2], facecolor = utils.COLORS['scorebase'], alpha = 0.2)

    ax.plot(xrange,st[0],c = utils.COLORS['stad'], label = 'STAD')
    ax.fill_between(xrange,st[1][0],st[1][2], facecolor = utils.COLORS['stad'], alpha = 0.2)


    ax.set_ylabel('Loss')
    ax.set_xlabel('Steps')
    ax.set_title('Design Optimization')
    ax.legend(loc = 'lower left')
