from drnet import doubly_robust
from drnet_em import doubly_robust_em
from autognet import evaluate_autognet_via_agc_effect
import numpy as np

column_names = ['average', 'direct_effect', 'spillover_effect', 'psi_1_gamma',
               'psi_0_gamma', 'psi_zero']

def run_dr(A_chain, L_chain, Y_chain, adj, i, mispec=None):
    """
    Run doubly robust estimator
    """
    ret_i = doubly_robust(A_chain[i], L_chain[i], Y_chain[i], adj, mispec=mispec)
    ret_array = np.array([ret_i[column_names[i]] for i in range(len(column_names))])
    # save results
    np.save(f'run/run_dr/drnet_{i}.npy', ret_array)
    
    return ret_array

def run_dr_em(A_chain, L_chain, Y_chain, adj, i):
    """
    Run doubly robust estimator
    """
    ret_i = doubly_robust_em(A_chain[i], L_chain[i], Y_chain[i], adj)
    ret_array = np.array([ret_i[column_names[i]] for i in range(len(column_names))])
    # save results
    np.save(f'run/run_dr_em/drnet_em_{i}.npy', ret_array)
    
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

cols_raw = ['psi_gamma', 'psi_zero', 'psi_1_gamma', 'psi_0_gamma']

def run_dr_raw(A_chain, L_chain, Y_chain, adj, i, treatment_allocation, psi_0_gamma_only, mispec=None):
    """
    Run doubly robust estimator
    """
    ret_i = doubly_robust(A_chain[i], L_chain[i], Y_chain[i], adj, treatment_allocation=treatment_allocation, seed=1, return_raw=True,
                          psi_0_gamma_only=psi_0_gamma_only, mispec=mispec)
    ret_array = np.zeros((ret_i[cols_raw[0]].shape[0], len(cols_raw)))
    for k in range(len(cols_raw)):
        ret_array[:, k] = ret_i[cols_raw[k]].copy()
    # save results
    np.save(f'run/run_dr_raw/drnet_raw_{i}.npy', ret_array)
    
    return ret_array

def run_dr_em_raw(A_chain, L_chain, Y_chain, adj, i, treatment_allocation):
    """
    Run doubly robust estimator
    """
    ret_i = doubly_robust_em(A_chain[i], L_chain[i], Y_chain[i], adj, treatment_allocation=treatment_allocation, seed=1, return_raw=True)
    ret_array = np.zeros((ret_i[cols_raw[0]].shape[0], len(cols_raw)))
    for k in range(len(cols_raw)):
        ret_array[:, k] = ret_i[cols_raw[k]].copy()
    # save results
    np.save(f'run/run_dr_em_raw/drnet_em_raw_{i}.npy', ret_array)
    
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