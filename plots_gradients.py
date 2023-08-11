import numpy as np
import seaborn as sns

def analyse_at_point(ax,runs1, eps=0.05, nsig = 50, nbins = 101, legend = True):
    values1 = np.array([r1['primal'] for r1 in runs1])
    
    stad_grads1 = np.array([r['grad_dict']['stad'] for r in runs1])
    score_grads1 = np.array([r['grad_dict']['score'] for r in runs1])
    dlogp = np.array([r['dlogp'] for r in runs1])
    numeric = np.array([r['grad_dict']['numeric'] for r in runs1])
    score_baseline = score_grads1 - dlogp*values1.mean()
    
    ax = sns.boxplot(ax = ax, data = np.column_stack(
        [stad_grads1,score_baseline, score_grads1, numeric]), orient = 'v', fliersize=1, meanline=True, showmeans=True,
        meanprops = {'c': 'k'}
    )
    ax.axhline(stad_grads1.mean(), c = 'k')


    
    return {'stad_m':stad_grads1.mean(), 'stad_s':stad_grads1.std(),
            'score_m':score_grads1.mean(), 'score_s':score_grads1.std(),
            'score_baseline_m':score_baseline.mean(), 'score_baseline_s':score_baseline.std(),
            'numeric_m':numeric.mean(), 'numeric_s':numeric.std(),           
           }


def plot_variance_with_inset(axarr,runs):
    ax = axarr
    _ = analyse_at_point(ax,runs, nsig = 50, nbins = 101, legend = True)
    ax.set_ylim(-15,15)
    ax.set_xticklabels(['StochAD','Score w/ Baseline','Score','Numeric'], rotation = 20)
    ax.set_xlim(-0.5,2.5)
    ax.set_ylabel(r'$g \sim \partial_\theta\mathbb{E}[X(\theta)]}$')


    iax = ax.inset_axes([0.3,0.1,0.3,0.2])
    _ = analyse_at_point(iax,runs, nsig = 50, nbins = 101, legend = True)
    iax.set_xticklabels(['ST','SB','Score','Numeric'])
    iax.set_xlim(1.5,3.5)
    iax.set_ylim(-60,60)    
