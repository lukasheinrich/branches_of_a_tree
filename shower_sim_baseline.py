# shower_sim_baseline

import numpy as np
import jax

def propagate_state(state):
    
    if state is None:
        return None
    
    if state['alive']==False:
        return state
        
    E,x,y,px,py = state['E'],state['x'],state['y'],state['px'],state['py']
    pmag = np.sqrt(px**2 + py**2)
    time = 0.02
    next_x = x + time*px
    next_y = y + time*py
    next_E = E
    return {
        'E': next_E,
        'x': next_x,
        'y': next_y,
        'px': state['px'],
        'py': state['py'],
        'alive':True,
    }
    
def sample_stop_prob(score, state, sim_parameters):
    r = np.sqrt(state['x']**2 + state['y']**2)
    E = state['E']
    par_thresh_E = sim_parameters['thresh_E']
    stop = False
    if (E < par_thresh_E):
        stop = True
    if (r > 20.):
        stop = True
    return stop

def interact_prob(x,y,par):
    par_radial = 10
    par_azimutal = 10
    r = jax.numpy.sqrt(x**2+y**2)

    alpha = jax.numpy.arctan2(x,y)


    sampling1 = 1/(1+jax.numpy.exp(10*jax.numpy.sin(par_radial*(alpha+2*r))))
    sampling2 = 1/(1+jax.numpy.exp(10*jax.numpy.cos(par_azimutal*(r-2))))
    start = 1/(1+jax.numpy.exp(-10*(r-par)))
    end = 1/(1+jax.numpy.exp(10*(r-(par+10.0))))

    return 0.5*start*sampling1*sampling2*end



########### Summary / Objective ###########

def per_hit_summary(hits):
    return np.sqrt(hits[:,0]**2+hits[:,1]**2)

def summary(generation):
    hits,active,*_ = generation
    return (np.mean(per_hit_summary(active)) - 2.0)**2

def summary_metric(active):
    return (np.mean(per_hit_summary(active)) - 2.0)**2
