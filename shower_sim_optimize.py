import numpy as np
import tqdm
import optax
import jax.numpy as jnp

def program_to_optimize(simulator, objective, sim_kwargs):
    def program_for_optimizer(theta, grad_type = "stad", eps=0.01, keep_all_grads=False):
        
        if grad_type not in ["stad","score","numeric"]:
            print("grad_type=",grad_type,"not recognized")
            return None
        
        hits,active,history,scores,out_st = simulator(theta, **sim_kwargs)
        primal = objective(active)
        
        grad_dict={}
        
        if grad_type == "stad" or keep_all_grads:
            alt = objective(out_st['y']['active'])
            grad_dict["stad"] = out_st['d'] + out_st['w']*(alt - primal)
        
        if grad_type == "score" or keep_all_grads:
            grad_dict["score"] = scores*primal
        
        if grad_type == "numeric" or keep_all_grads:
            _,active2,_,_,_ = simulator(theta+eps, **sim_kwargs)
            primal2 = objective(active2)
            grad_dict["numeric"] = (primal2 - primal) / eps
        
        grad_val = grad_dict[grad_type]
        
        return {"primal":primal, "grad":grad_val, "grad_type":grad_type, "grad_dict":grad_dict, "dlogp":scores}
    return program_for_optimizer

def minibatch_primal_and_grad(program, theta, Nmini, grad_type = "stad", dobaseline=True):
    
    runs = [program(theta, grad_type) for _ in range(Nmini)]
    
    primal = np.mean([r["primal"] for r in runs])
    
    if grad_type=="score" and dobaseline and Nmini > 1:
        grad = np.mean([ (r["grad"]-r["dlogp"]*primal) for r in runs])
    else:
        grad = np.mean([r["grad"] for r in runs])
 
    
    return primal, grad

def optimize(program, init, LR, Nepoch, Nmini, grad_type, dobaseline=True, doprint=True):
    traj_theta = []
    traj_v = []
    traj_g = []
    theta = jnp.array(init)
    
    optimizer = optax.adam(learning_rate=LR)
    adam_state = optimizer.init(theta)

    trainsteps = tqdm.tqdm(range(Nepoch))
    for i in trainsteps:
        traj_theta.append(theta)
        v, g = minibatch_primal_and_grad(program, theta, Nmini, grad_type, dobaseline)
        updates, adam_state = optimizer.update(g, adam_state, theta)
        theta = optax.apply_updates(theta, updates)
        if theta < 0.:
            theta = 0.
        
        traj_v.append(v)
        traj_g.append(g)
    return theta, traj_theta, traj_v, traj_g
