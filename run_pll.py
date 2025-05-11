from drnet import doubly_robust
from autognet import evaluate_autognet_via_agc_effect
from ocnet import ocnet
import numpy as np

column_names = ['average', 'direct_effect', 'spillover_effect', 
               'psi_0_gamma', 'psi_zero', 'psi_1_gamma']

def run_dr(A_chain, L_chain, Y_chain, adj, i):
    """
    Run doubly robust estimator
    """
    ret_i = doubly_robust(A_chain[i], L_chain[i], Y_chain[i], adj)
    ret_array = np.array([ret_i[column_names[i]] for i in range(len(column_names))])
    # save results
    np.save(f'run/run_dr/drnet_{i}.npy', ret_array)
    
    return ret_array


def run_autognet(A_chain, L_chain, Y_chain, adj, i):
    """
    Run autognet estimator
    """
    ret_i = evaluate_autognet_via_agc_effect(adj, Y_chain[i], A_chain[i], L_chain[i], treatment_allocation=0.7, 
                                             R=50, burnin=10, seed=1)
    ret_array = np.array([ret_i[column_names[i]] for i in range(len(column_names))])
    # save results
    np.save(f'run/run_autog/autognet_{i}.npy', ret_array)
    
    return ret_array

def run_ocnet(A_chain, L_chain, Y_chain, adj, i):
    """
    Run ocnet estimator
    """
    ret_i = ocnet(A_chain[i], L_chain[i], Y_chain[i], adj, treatment_allocation=0.7, num_rep=1000, seed=1)
    ret_array = np.array([ret_i[column_names[i]] for i in range(len(column_names))])
    # save results
    np.save(f'run/run_oc/ocnet_{i}.npy', ret_array)
    
    return ret_array

cols_raw = ['psi_gamma', 'psi_zero', 'psi_1_gamma', 'psi_0_gamma']

def run_dr_raw(A_chain, L_chain, Y_chain, adj, i, treatment_allocation, psi_0_gamma_only):
    """
    Run doubly robust estimator
    """
    ret_i = doubly_robust(A_chain[i], L_chain[i], Y_chain[i], adj, treatment_allocation=treatment_allocation, seed=1, return_raw=True,
                          psi_0_gamma_only=psi_0_gamma_only)
    ret_array = np.zeros((ret_i[cols_raw[0]].shape[0], ret_i[cols_raw[0]].shape[1], len(cols_raw)))
    for i in range(len(cols_raw)):
        ret_array[:, :, i] = ret_i[cols_raw[i]].copy()
    # save results
    np.save(f'run/run_dr_raw/drnet_raw_{i}.npy', ret_array)
    
    return ret_array

def run_autognet_raw(A_chain, L_chain, Y_chain, adj, i, treatment_allocation):
    """
    Run autognet estimator
    """
    ret_i = evaluate_autognet_via_agc_effect(adj, Y_chain[i], A_chain[i], L_chain[i], treatment_allocation=treatment_allocation, 
                                             R=50, burnin=10, seed=1)
    ret_array = np.array([ret_i[column_names[i]] for i in range(len(column_names))])
    # save results
    np.save(f'run/run_autog_raw/autognet_raw_{i}.npy', ret_array)
    
    return ret_array