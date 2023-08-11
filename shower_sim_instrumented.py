import numpy as np
import jax
import copy
import queue
from shower_sim_baseline import interact_prob, propagate_state, sample_stop_prob

def stochasticTriple(d=0., y=None, w=0.):
    return {
        "d": d,
        "y": y,
        "w": w,
    }


def bernoulli_basic(p, get_omega=False, u_input=None):
    def result_on_omega(omega,p):
        if omega > (1.-p):
            return 1
        else:
            return 0
    
    if u_input is not None:
        u = u_input
    else:
        u = np.random.uniform()
        
    b = result_on_omega(u,p)
        
    if get_omega:
        return b, u
    else:
        return b

def bernoulli(p, p_st, get_omega=False):
        
    def _fwd(p, get_omega=False):    
        def result_on_omega(omega,p):
            if omega > (1.-p):
                return 1
            else:
                return 0
            
        u = np.random.uniform()
        b = result_on_omega(u,p)
        
        if get_omega:
            return b, u
        else:
            return b
        
        
    def _deriv(x,p, direction=1.0):
        if int(direction)==1:
            if x==0:
                #Right deriv
                st = stochasticTriple(0., 1, 1./(1.-p))
            else:
                st = stochasticTriple(0., 0, 0.)
                
        else:
            if x == 1: 
                #Left deriv
                st = stochasticTriple(0., 0, 1./p)
            else:
                st = stochasticTriple(0., 0, 0.)      
            
        return st
    
    if get_omega:
        out, u = _fwd(p, get_omega=True)
    else:
        out = _fwd(p, get_omega=False)
    
    direction=1.0  #by default, choose right deriv, but switch if p deriv is negative
    if p_st is not None:
        if p_st['d'] < 0:
            direction = -1.0
    
    out_st = _deriv(out, p, direction)
    
    if p_st is not None: 
        out_st = compose_derivs(_fwd, p_st, out_st)
    
    if get_omega:
        return out, out_st, u
    else:
        return out, out_st        
    
def compose_derivs(func,st1,st2):
    d1,y1,w1 = st1["d"], st1["y"], st1["w"]
    d2,y2,w2 = st2["d"], st2["y"], st2["w"]
    
    w1_iszero = (w1==0 or w1==0.)
    w2_iszero = (w2==0 or w2==0.)
    
    u = None
    
    if w1_iszero and w2_iszero:
        y=y2
    else:
        prob = 0 if w1_iszero else np.fabs(w1)/(np.fabs(w1)+np.fabs(d1)*np.fabs(w2))
        option = bernoulli_basic(prob, get_omega=False)
        
        y = func(y1) if option == 1 else y2
        
    d = d1*d2
    w = np.fabs(w1) + np.fabs(d1)*np.fabs(w2)
    
    #print("C", st1, st2, (d,y,w))
    
    return stochasticTriple(d,y,w)


def do_prune_away_old(st_new, st_old):
    
    w_new = st_new["w"]
    w_old = st_old["w"]
    
    w_new_iszero = (w_new==0 or w_new==0.)
    w_old_iszero = (w_old==0 or w_old==0.)
    
    
    if w_new_iszero and w_old_iszero:
        keep_new_state = False
    else:
        prob = 0 if w_new_iszero else np.fabs(w_new)/(np.fabs(w_new)+np.fabs(w_old))
        keep_new_state = bernoulli_basic(prob, get_omega=False)

    return keep_new_state

score_bernoulli = jax.jit(jax.grad(jax.scipy.stats.bernoulli.logpmf, argnums=1))
score_bernoulli(1,0.5)



interact_prob_and_g = jax.jit(jax.value_and_grad(interact_prob, argnums = 2))
interact_prob_and_g(0.5,0.5,0.5)

def interact_prob_and_grad(x,y,par):
    p,g = interact_prob_and_g(x,y,par)
    
    g = jax.lax.cond(jax.numpy.isnan(g), lambda x: 0., lambda x: x, g)
    #if jax.numpy.isnan(g):
    #    g=0.
    
    return p, g

interact_prob_and_grad = jax.jit(interact_prob_and_grad)
interact_prob_and_grad(0.5,0.5,0.5)


def sample_interact(score, state, sim_parameters, keep_derivs=True, fifos=None):
    x,y = state['x'], state['y']
    par_thresh_x = sim_parameters['thresh_x']

    interact_prob, interact_prob_grad = interact_prob_and_grad(x,y,par_thresh_x)
    
    if not keep_derivs:
        if fifos is not None and not fifos['interact'].empty():
            u_input = fifos['interact'].get()
            interact = bernoulli_basic(interact_prob, get_omega=False, u_input=u_input)
        else:
            interact = bernoulli_basic(interact_prob, get_omega=False, u_input=None)
            
        return interact, None
        
    interact_prob_st = stochasticTriple(interact_prob_grad, 0., 0.)
    
    if fifos is not None:
        interact, interact_st, u_int = bernoulli(interact_prob, interact_prob_st, get_omega=True)
        fifos['interact'].put(u_int)
    else:
        interact, interact_st = bernoulli(interact_prob, interact_prob_st, get_omega=False)
    
    score['thresh_x'] += score_bernoulli(interact,interact_prob)*interact_prob_grad

    return interact, interact_st

def sample_fate(score, state, program_st, sim_parameters, check_alts=True, fifos=None):
    stop = sample_stop_prob(score, state,sim_parameters)
    if stop:
        state['alive']=False
        return None
    
    if check_alts==False:
        interact, _ = sample_interact(None, state, sim_parameters, keep_derivs=False, fifos=fifos)
        interact_st = None
        keep_new_alt = False
    else:
        interact, interact_st = sample_interact(score, state, sim_parameters, keep_derivs=True, fifos=fifos)
        
        keep_new_alt = False
        if program_st['y']==None:
            keep_new_alt = True
        else:
            keep_new_alt = do_prune_away_old(interact_st, program_st)
        
        
        #keep_new_alt = True if program_st['y']==None else do_prune_away_old(interact_st, program_st)

        program_st['w'] += np.fabs(interact_st['w'])
        if keep_new_alt:
            program_st['d'] = interact_st['d']
            
            if fifos is not None and not fifos['interact'].empty():
                #_ = fifos['interact'].get() #remove this rv, as it was used to creat alternative
                fifos['interact'].queue.clear()
    
    
    bumpx = 0.05
    bumpy = 0.05#np.random.normal(0,.1)
    
    split = np.random.binomial(1, sim_parameters['split_prob'])
    
    if not interact:
        return {
            'interact': False, 
            'interact_st': interact_st, 
            'keep_new_alt': keep_new_alt,
            'split': split,
            'eloss': 2.0,
            'bumpx': bumpx, 
            'bumpy': bumpy,
        }


    
    if split:
        return {
            'interact': True, 
            'interact_st': interact_st, 
            'keep_new_alt': keep_new_alt,
            'split': True, 
            'eloss': 2.0,
            'bumpx': bumpx, 
            'bumpy': bumpy
        }
    else:
        return {
            'interact': True, 
            'interact_st': interact_st, 
            'keep_new_alt': keep_new_alt,
            'split': False, 
            'eloss': 2.0,
            'bumpx': bumpx, 
            'bumpy': bumpy
        }#np.random.uniform(0,0.1)}

def fate2state(fate, state):
    def _update_stop(state):
        state1 = {
            'E': state['E'],
            'x': state['x'],
            'y': state['y'],
            'px': state['px'] + fate['bumpx'],
            'py': state['py'] + fate['bumpy'],
            'alive':False,
          }
        return state1
    
    def _update_split(state):
        
        norm1 = np.sqrt( (state['px'] + fate['bumpx'])**2 + (state['py'] + fate['bumpy'])**2 )
        norm2 = np.sqrt( (state['px'] - fate['bumpx'])**2 + (state['py'] - fate['bumpy'])**2 )

        
        state1 = {
            'E': state['E']/2,
            'x': state['x'],
            'y': state['y'],
            'px': (state['px'] + fate['bumpx']) / norm1,
            'py': (state['py'] + fate['bumpy']) / norm1,
            'alive':True,
          }
        state2 = {
            'E': state['E']/2,
            'x': state['x'],
            'y': state['y'],
            'px': (state['px'] - fate['bumpx']) / norm2,
            'py': (state['py'] - fate['bumpy']) / norm2,
            'alive':True,
          }
        return state1, state2
    
    def _update_eloss(state):
        new_E = state['E'] - fate['eloss']
        if new_E < 0.:
            new_E = 0.
            
        bump_px = state['px'] + fate['bumpx']*(1.0 if np.random.binomial(1,0.5) else -1.0)
        bump_py = state['py'] + fate['bumpy']*(1.0 if np.random.binomial(1,0.5) else -1.0)
                
        #renomalize 
        new_px = bump_px / np.sqrt(bump_px**2 + bump_py**2)
        new_py = bump_py / np.sqrt(bump_px**2 + bump_py**2)
        
        state1 = {
            'E': new_E,
            'x': state['x'],
            'y': state['y'],
            'px': new_px,
            'py': new_py,
            'alive':True,
          }
        return state1
    
    
    if fate is None:
        state1 = _update_stop(state)
        return state1, None, None, None
    
    if not fate['interact']:
        state1, state2 = state, None
        
        stateY1, stateY2 = None, None
        if fate['keep_new_alt']:
            if fate['split']:
                stateY1, stateY2 = _update_split(state)
            else:
                stateY1 = _update_eloss(state)
                stateY2 = None
        
        return state1,state2,stateY1,stateY2
        
    
    if fate['split']:        
        state1, state2 = _update_split(state)
        
        stateY1, stateY2 = None, None
        if fate['keep_new_alt']:
            #stateY1 = _update_eloss(state)
            stateY1, stateY2 = state, None
            
        return state1,state2,stateY1,stateY2
        
    else:
        state1 = _update_eloss(state)
        state2 = None
        
        stateY1, stateY2 = None, None
        if fate['keep_new_alt']:
            #stateY1, stateY2 = _update_split(state)
            stateY1, stateY2 = state, None
            
    
        return state1,state2,stateY1,stateY2


def run(score, history, hits, alive_states, program_st, sim_parameters, step_count, fifos = None):
    next_alive_states = []
    next_alive_states_st = []
    found_alt = False
    
    for state in alive_states:
        # we loop over all particles that are still activee


        # first do a deterministic propagation to a new position
        new_state = propagate_state(state)

        #then 
        fate = sample_fate(score, new_state, program_st, sim_parameters, fifos = fifos)
        
        if fate is None:
            hits.append([state['x'],state['y'],state['E'],1])
        else:
            hits.append([state['x'],state['y'],state['E'],fate['interact']])
            history.append([[state['x'],state['y']],[new_state['x'],new_state['y']]])
            next1, next2, nextY1, nextY2 = fate2state(fate, new_state)
            
            #print(step_count['n'], fate['keep_new_alt'])
            
            if fate['keep_new_alt']==True:
                found_alt = True
                next_alive_states_st = copy.deepcopy(next_alive_states)                
                if nextY1 is not None:
                    next_alive_states_st.append(nextY1)
                if nextY2 is not None:
                    next_alive_states_st.append(nextY2)
            
            
            if next1 is not None:
                next_alive_states.append(next1)
                if found_alt and fate['keep_new_alt']==False:
                    next_alive_states_st.append(next1)
                    
            if next2 is not None:
                next_alive_states.append(next2)
                if found_alt and fate['keep_new_alt']==False:
                    next_alive_states_st.append(next2)
                    
    if found_alt:
        program_st['y'] = {
            'history': copy.deepcopy(history),
            'hits': copy.deepcopy(hits),
            'alive_states': next_alive_states_st,
        }
        
                    
    else:
        if program_st['y'] is not None:
            next_alive_states_st = []

            for state in program_st['y']['alive_states']:
                new_state = propagate_state(state)
                fate = sample_fate(None, new_state, None, sim_parameters, check_alts=False, fifos = fifos)
            
                if fate is None:
                    program_st['y']['hits'].append([state['x'],state['y'],state['E'],1])
                else:
                    program_st['y']['hits'].append([state['x'],state['y'],state['E'],fate['interact']])
                    program_st['y']['history'].append([[state['x'],state['y']],[new_state['x'],new_state['y']]])
                    next1, next2, _, _ = fate2state(fate, new_state)
                    
                    
                    if next1 is not None:
                        next_alive_states_st.append(next1)
                    if next2 is not None:
                        next_alive_states_st.append(next2)
            
            program_st['y']['alive_states'] = next_alive_states_st
            
    run_again = (len(next_alive_states)>0)
    run_again_st = False
    if (program_st['y'] is None):  
        run_again_st = False 
    else:
        if(len(program_st['y']['alive_states'])>0):
            run_again_st = True
 
    step_count['n'] += 1
    
    if run_again or run_again_st:
        #print(len(next_alive_states))
        try:
            run(score, history, hits, next_alive_states, program_st, sim_parameters, step_count, fifos)
        except RecursionError as e:
            print("####### Caught Recursion Error #######")
            print(len(next_alive_states), len(program_st['y']['alive_states']), step_count['n'])
            print("next_alive_states:")
            for state in next_alive_states:
                print(state)
                
            print("next_alive_states_st:")
            for state in program_st['y']['alive_states']:
                print(state)
                
            return


def generate_random_init():
    phi = np.random.uniform(-np.pi,np.pi)
    py = np.sin(phi)
    px = np.cos(phi)
    state = {'x': 0, 'y': 0, 'r': 0, 'px': px, 'py': py, 'E': 25, 'alive':True}
    return state


def generate(init, program_st, sim_parameters, reuse_rvs=False):
    history, hits, score = [], [], {'thresh_x': 0.0, 'thresh_E': 0.0}
    step_count = {'n':0}
    
    fifos = None
    if reuse_rvs:
        fifos = { "interact":queue.Queue() }
        
    run(score, history, hits, alive_states = [init], 
        program_st = program_st, sim_parameters=sim_parameters, 
        step_count=step_count, fifos=fifos)
    
    hits = np.array(hits)
    act = hits[hits[:,3]==1]
    hits_st = np.array(program_st['y']['hits'])
    out_st = stochasticTriple(program_st['d'], 
                              {'hits':hits_st,
                               'active':hits_st[hits_st[:,3]==1],
                               'history':np.array(program_st['y']['history'])},
                             program_st['w'])
    return np.array(hits), act, np.array(history), {k:np.array(v) for k,v in score.items()}, out_st

def simulator(par, reuse_rvs = True):
    init = generate_random_init()
    program_st = stochasticTriple(0., None, 0.)
    
    sim_parameters = {'thresh_E': 0.5, 'split_prob':1.0, 'thresh_x': par}
    
    hits,active,history,scores,out_st = generate(init, program_st, sim_parameters, reuse_rvs)
    scores = jax.lax.stop_gradient(scores['thresh_x'])
    return hits,active,history,scores,out_st                    